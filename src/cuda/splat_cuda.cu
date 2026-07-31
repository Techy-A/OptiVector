// CurveSplat CUDA kernel: per-curve Gaussian-splat ALPHA (coverage * opacity).
//
// Wraps only the expensive geometry (params -> alpha[C,H,W]); the ordered
// alpha-over composite stays in PyTorch (cheap cumprod). Replicates the math in
// src/python/curvesplat/renderer.py::render_curves_batched EXACTLY so the
// pure-PyTorch path remains the numerical reference / fallback.
//
// Per curve c, per pixel (x,y): loop N Bezier samples; oriented 2D Gaussian
// (sigma_long=width, sigma_short=width/4); 3-sigma cutoff (mahal<=9);
// coverage = sum_n exp(-0.5*mahal); alpha = clamp(coverage/N,0,1) * opacity.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define DET_EPS 1e-4f
#define MAHAL_CUTOFF 9.0f

// ---- device helpers: cubic Bezier basis (position + derivative) ----
__device__ __forceinline__ void bezier_basis(float t, float* B) {
    float omt = 1.0f - t;
    B[0] = omt*omt*omt;
    B[1] = 3.0f*omt*omt*t;
    B[2] = 3.0f*omt*t*t;
    B[3] = t*t*t;
}
__device__ __forceinline__ void bezier_dbasis(float t, float* Bd) {
    float omt = 1.0f - t;
    Bd[0] = -3.0f*omt*omt;
    Bd[1] = 3.0f*omt*omt - 6.0f*omt*t;
    Bd[2] = 6.0f*omt*t - 3.0f*t*t;
    Bd[3] = 3.0f*t*t;
}

// Inverse covariance entries (iXX,iXY,iYY) from tangent + width. Mirrors
// build_covariance + invert_2x2 (closed-form 2x2, det guarded to DET_EPS).
__device__ __forceinline__ void inv_cov(
    float tanx, float tany, float w,
    float& iXX, float& iXY, float& iYY,
    float& c2, float& s2, float& sl, float& ss, float& det) {
    float norm = sqrtf(tanx*tanx + tany*tany + 1e-12f);
    float dxs, dys;
    if (norm < 1e-6f) { dxs = 1.0f; dys = 0.0f; }
    else { dxs = tanx; dys = tany; }
    float theta = (norm < 1e-6f) ? 0.0f : atan2f(dys, dxs);
    c2 = cosf(theta); s2 = sinf(theta);
    sl = w*w;
    ss = (w/4.0f)*(w/4.0f);
    float Sxx = c2*c2*sl + s2*s2*ss;
    float Sxy = c2*s2*(sl - ss);
    float Syy = s2*s2*sl + c2*c2*ss;
    det = Sxx*Syy - Sxy*Sxy;
    if (fabsf(det) < DET_EPS) det = DET_EPS;
    iXX = Syy/det;
    iXY = -Sxy/det;
    iYY = Sxx/det;
}

// ============================ FORWARD ============================
__global__ void alpha_forward_kernel(
    const float* __restrict__ cp,       // (C,4,2)
    const float* __restrict__ width,    // (C,)
    const float* __restrict__ opacity,  // (C,)
    float* __restrict__ alpha,          // (C,H,W) out
    int C, int H, int W, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)C * H * W;
    if (idx >= total) return;

    int c = idx / (H*W);
    int rem = idx - c*(H*W);
    int y = rem / W;
    int x = rem - y*W;

    float gx = (float)x + 0.5f;
    float gy = (float)y + 0.5f;
    float w = width[c];
    float op = opacity[c];
    const float* cpc = cp + c*8;  // 4*2

    float cov = 0.0f;
    float invN1 = (N > 1) ? 1.0f/(float)(N-1) : 0.0f;
    for (int n = 0; n < N; ++n) {
        float t = (N > 1) ? (float)n * invN1 : 0.0f;
        float B[4], Bd[4];
        bezier_basis(t, B); bezier_dbasis(t, Bd);
        float px=0, py=0, tx=0, ty=0;
        for (int i = 0; i < 4; ++i) {
            px += B[i]*cpc[i*2+0];  py += B[i]*cpc[i*2+1];
            tx += Bd[i]*cpc[i*2+0]; ty += Bd[i]*cpc[i*2+1];
        }
        float iXX,iXY,iYY,c2,s2,sl,ss,det;
        inv_cov(tx,ty,w,iXX,iXY,iYY,c2,s2,sl,ss,det);
        float dx = gx - px, dy = gy - py;
        float mahal = iXX*dx*dx + 2.0f*iXY*dx*dy + iYY*dy*dy;
        if (mahal <= MAHAL_CUTOFF) {
            float m = fminf(mahal, 50.0f);
            cov += expf(-0.5f*m);
        }
    }
    float covn = cov / (float)N;
    if (covn > 1.0f) covn = 1.0f;
    if (covn < 0.0f) covn = 0.0f;
    alpha[idx] = covn * op;
}

// ============================ BACKWARD ============================
// Given grad_alpha (C,H,W) -> grad_cp (C,4,2), grad_width (C), grad_opacity (C).
__global__ void alpha_backward_kernel(
    const float* __restrict__ grad_alpha,
    const float* __restrict__ cp,
    const float* __restrict__ width,
    const float* __restrict__ opacity,
    float* __restrict__ grad_cp,        // (C,4,2)
    float* __restrict__ grad_width,     // (C,)
    float* __restrict__ grad_opacity,   // (C,)
    int C, int H, int W, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)C * H * W;
    if (idx >= total) return;

    int c = idx / (H*W);
    int rem = idx - c*(H*W);
    int y = rem / W;
    int x = rem - y*W;

    float gA = grad_alpha[idx];
    if (gA == 0.0f) return;

    float gx = (float)x + 0.5f;
    float gy = (float)y + 0.5f;
    float w = width[c];
    float op = opacity[c];
    const float* cpc = cp + c*8;
    float invN1 = (N > 1) ? 1.0f/(float)(N-1) : 0.0f;
    float invN  = 1.0f/(float)N;

    // Pass 1: coverage (to apply clamp derivative).
    float cov = 0.0f;
    for (int n = 0; n < N; ++n) {
        float t = (N > 1) ? (float)n * invN1 : 0.0f;
        float B[4], Bd[4]; bezier_basis(t,B); bezier_dbasis(t,Bd);
        float px=0,py=0,tx=0,ty=0;
        for (int i=0;i<4;++i){px+=B[i]*cpc[i*2];py+=B[i]*cpc[i*2+1];tx+=Bd[i]*cpc[i*2];ty+=Bd[i]*cpc[i*2+1];}
        float iXX,iXY,iYY,c2,s2,sl,ss,det; inv_cov(tx,ty,w,iXX,iXY,iYY,c2,s2,sl,ss,det);
        float dx=gx-px, dy=gy-py;
        float mahal = iXX*dx*dx + 2.0f*iXY*dx*dy + iYY*dy*dy;
        if (mahal <= MAHAL_CUTOFF) cov += expf(-0.5f*fminf(mahal,50.0f));
    }
    float covn = cov*invN;
    float covn_clamped = fminf(fmaxf(covn,0.0f),1.0f);

    // alpha = covn_clamped * op
    atomicAdd(&grad_opacity[c], gA * covn_clamped);

    // d alpha / d cov = (op/N) inside the clamp's active region, else 0.
    if (covn <= 0.0f || covn >= 1.0f) return;
    float gcov = gA * op * invN;   // dL/d(cov) = dL/dalpha * op * (1/N)

    // Pass 2: distribute gcov through each sample's Gaussian to params.
    for (int n = 0; n < N; ++n) {
        float t = (N > 1) ? (float)n * invN1 : 0.0f;
        float B[4], Bd[4]; bezier_basis(t,B); bezier_dbasis(t,Bd);
        float px=0,py=0,tx=0,ty=0;
        for (int i=0;i<4;++i){px+=B[i]*cpc[i*2];py+=B[i]*cpc[i*2+1];tx+=Bd[i]*cpc[i*2];ty+=Bd[i]*cpc[i*2+1];}
        float iXX,iXY,iYY,c2,s2,sl,ss,det; inv_cov(tx,ty,w,iXX,iXY,iYY,c2,s2,sl,ss,det);
        float dx=gx-px, dy=gy-py;
        float mahal = iXX*dx*dx + 2.0f*iXY*dx*dy + iYY*dy*dy;
        if (mahal > MAHAL_CUTOFF) continue;
        float weight = expf(-0.5f*fminf(mahal,50.0f));
        // dL/dmahal_n  (clamp(mahal,50): deriv 1 since mahal<=9<50)
        float dmahal = gcov * (-0.5f*weight);

        // --- mahal -> position (cp via position basis B) ---
        float dm_dpx = -(2.0f*iXX*dx + 2.0f*iXY*dy);
        float dm_dpy = -(2.0f*iYY*dy + 2.0f*iXY*dx);

        // --- mahal -> inv-cov entries ---
        float dm_diXX = dx*dx;
        float dm_diXY = 2.0f*dx*dy;
        float dm_diYY = dy*dy;

        // inv-cov -> (Sxx,Sxy,Syy) -> (theta, width). Recompute S + det.
        float Sxx = c2*c2*sl + s2*s2*ss;
        float Sxy = c2*s2*(sl - ss);
        float Syy = s2*s2*sl + c2*c2*ss;
        float det2 = det*det;
        // helper: given dSxx,dSxy,dSyy -> dmahal via inv entries
        // ddet = dSxx*Syy + Sxx*dSyy - 2*Sxy*dSxy
        // diXX = (dSyy*det - Syy*ddet)/det^2
        // diXY = (-dSxy*det + Sxy*ddet)/det^2
        // diYY = (dSxx*det - Sxx*ddet)/det^2
        #define DMAHAL_FROM_S(dSxx,dSxy,dSyy) ( \
            dm_diXX*((dSyy)*det - Syy*((dSxx)*Syy+Sxx*(dSyy)-2.0f*Sxy*(dSxy)))/det2 + \
            dm_diXY*(-(dSxy)*det + Sxy*((dSxx)*Syy+Sxx*(dSyy)-2.0f*Sxy*(dSxy)))/det2 + \
            dm_diYY*((dSxx)*det - Sxx*((dSxx)*Syy+Sxx*(dSyy)-2.0f*Sxy*(dSxy)))/det2 )

        // width derivatives of S: sl'=2w, ss'=2*(w/16)=w/8
        float dsl_dw = 2.0f*w;
        float dss_dw = w/8.0f;
        float dSxx_dw = c2*c2*dsl_dw + s2*s2*dss_dw;
        float dSxy_dw = c2*s2*(dsl_dw - dss_dw);
        float dSyy_dw = s2*s2*dsl_dw + c2*c2*dss_dw;
        float dm_dw = DMAHAL_FROM_S(dSxx_dw, dSxy_dw, dSyy_dw);
        atomicAdd(&grad_width[c], dmahal * dm_dw);

        // theta derivatives of S
        float dSxx_dth = 2.0f*c2*s2*(ss - sl);
        float dSxy_dth = (c2*c2 - s2*s2)*(sl - ss);
        float dSyy_dth = 2.0f*c2*s2*(sl - ss);
        float dm_dth = DMAHAL_FROM_S(dSxx_dth, dSxy_dth, dSyy_dth);

        // theta -> tangent (atan2), tangent -> cp via derivative basis.
        float tnorm2 = tx*tx + ty*ty;
        float dth_dtx = 0.0f, dth_dty = 0.0f;
        if (tnorm2 > 1e-12f) { dth_dtx = -ty/tnorm2; dth_dty = tx/tnorm2; }

        for (int i = 0; i < 4; ++i) {
            // position path (mahal -> pos -> cp via B)
            float gcx = dmahal * dm_dpx * B[i];
            float gcy = dmahal * dm_dpy * B[i];
            // theta path (mahal -> theta -> tan -> cp via Bd)
            gcx += dmahal * dm_dth * dth_dtx * Bd[i];
            gcy += dmahal * dm_dth * dth_dty * Bd[i];
            atomicAdd(&grad_cp[c*8 + i*2 + 0], gcx);
            atomicAdd(&grad_cp[c*8 + i*2 + 1], gcy);
        }
        #undef DMAHAL_FROM_S
    }
}

// ============================ launchers ============================
torch::Tensor alpha_forward(torch::Tensor cp, torch::Tensor width,
                            torch::Tensor opacity, int64_t H, int64_t W, int64_t N) {
    auto C = cp.size(0);
    auto alpha = torch::zeros({C, H, W}, cp.options());
    long total = (long)C*H*W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    alpha_forward_kernel<<<blocks, threads>>>(
        cp.data_ptr<float>(), width.data_ptr<float>(), opacity.data_ptr<float>(),
        alpha.data_ptr<float>(), (int)C, (int)H, (int)W, (int)N);
    return alpha;
}

std::vector<torch::Tensor> alpha_backward(torch::Tensor grad_alpha, torch::Tensor cp,
                                          torch::Tensor width, torch::Tensor opacity, int64_t N) {
    auto C = cp.size(0);
    auto H = grad_alpha.size(1);
    auto W = grad_alpha.size(2);
    auto grad_cp = torch::zeros_like(cp);
    auto grad_width = torch::zeros_like(width);
    auto grad_opacity = torch::zeros_like(opacity);
    long total = (long)C*H*W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    alpha_backward_kernel<<<blocks, threads>>>(
        grad_alpha.contiguous().data_ptr<float>(), cp.data_ptr<float>(),
        width.data_ptr<float>(), opacity.data_ptr<float>(),
        grad_cp.data_ptr<float>(), grad_width.data_ptr<float>(), grad_opacity.data_ptr<float>(),
        (int)C, (int)H, (int)W, (int)N);
    return {grad_cp, grad_width, grad_opacity};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("alpha_forward", &alpha_forward, "CurveSplat alpha forward (CUDA)");
    m.def("alpha_backward", &alpha_backward, "CurveSplat alpha backward (CUDA)");
}
