---
name: svg-ai-brand-guidelines
description: Applies brand colors and typography (dark theme with neon lime accent) to artifacts, presentations, and UI mockups. Use when brand colors, style guidelines, visual formatting, or company design standards apply to SVG AI content.
---

# SVG AI Brand Styling

## Overview
This skill captures the visual identity — a high-contrast dark theme with a single neon accent color, used for AI product marketing and dashboard surfaces.

**Keywords**: branding, dark theme, neon accent, visual identity, styling, SVG AI brand, visual formatting, visual design

## Brand Guidelines

### Colors

**Main Colors:**
- Background Black: `#0A0A0A` - Primary page background
- Card Surface: `#1A1A1A` - Cards, panels, secondary surfaces
- White: `#FFFFFF` - Primary text and headlines on dark
- Mid Gray: `#A0A0A0` - Secondary/body text, descriptions

**Accent Colors:**
- Neon Lime: `#D7FE3D` - Primary accent (CTAs, highlights, key headings)
- Used sparingly — one bright accent against a near-black field for maximum contrast

### Typography
- **Headings**: Bold sans-serif (Inter / Helvetica Neue / Arial fallback) — large scale, tight tracking
- **Body Text**: Regular sans-serif (Inter / Arial / system-ui fallback)
- **Note**: All text in this theme is sans-serif; no serif pairing is used

## Features

### Smart Color Application
- Backgrounds default to near-black (`#0A0A0A`)
- Headlines and key phrases use Neon Lime (`#D7FE3D`) for emphasis — typically one phrase per section, not entire blocks
- Body/description text uses Mid Gray (`#A0A0A0`) for hierarchy against white headings
- Primary CTAs: Neon Lime fill with black/dark text, pill-shaped (high border-radius), often with a glow/halo effect
- Secondary CTAs: transparent/outlined buttons with Neon Lime border and white or lime text

### Text Styling
- Hero headings: White, large (40px+), bold
- Section labels (e.g. card category tags): White, uppercase, bold, smaller scale
- Body/description: Mid Gray, regular weight, comfortable line-height
- Links/nav items: White or light gray, medium weight

### Shape and Accent Usage
- Cards use dark gray (`#1A1A1A`) surfaces on the black background, subtle separation rather than borders
- Neon Lime is reserved for: primary buttons, key headline words, active/selected states, and small UI accents (e.g. badges, outlines on focused elements)
- Avoid overusing the accent color — it should "pop" against large areas of black/dark gray and white text

## Technical Details

### Color Application
- Use RGB/hex values exactly as specified for fidelity: background `#0A0A0A`, surfaces `#1A1A1A`, text white `#FFFFFF`, secondary text `#A0A0A0`, accent `#D7FE3D`
- Maintain strong contrast ratios — white/lime on near-black backgrounds only; avoid placing lime text on light backgrounds
- When generating UI mockups or slides in this theme, default every background to `#0A0A0A` or `#1A1A1A` and apply Neon Lime only to interactive or emphasis elements

### Font Management
- Use Inter as the primary sans-serif if available
- Fallback to Arial / Helvetica Neue / system-ui for both headings and body text
- No serif fonts are used anywhere in this theme
