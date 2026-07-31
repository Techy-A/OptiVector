import React, { useState, useEffect } from 'react';

interface Point { x: number; y: number; }
interface CurveData {
  id: number;
  startX: number; startY: number;
  cp1X: number; cp1Y: number;
  cp2X: number; cp2Y: number;
  endX: number; endY: number;
  stroke: string | null;
  strokeWidth: string | null;
  opacity: string | null;
}

interface InteractiveSVGProps {
  svgString: string;
}

export const InteractiveSVG: React.FC<InteractiveSVGProps> = ({ svgString }) => {
  const [curves, setCurves] = useState<CurveData[]>([]);
  const [selectedCurve, setSelectedCurve] = useState<number | null>(null);
  const [draggingNode, setDraggingNode] = useState<{curveId: number, nodeType: string} | null>(null);

  useEffect(() => {
    // Parse the raw SVG string
    const parser = new DOMParser();
    const doc = parser.parseFromString(svgString, "image/svg+xml");
    const paths = Array.from(doc.querySelectorAll("path"));
    
    const parsedCurves: CurveData[] = [];
    paths.forEach((p, i) => {
      const d = p.getAttribute("d");
      if (!d) return;
      // Expecting format: "M x y C x y x y x y"
      const match = d.match(/M\s+([-\d.]+)\s+([-\d.]+)\s+C\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)/);
      if (match) {
        parsedCurves.push({
          id: i,
          startX: parseFloat(match[1]), startY: parseFloat(match[2]),
          cp1X: parseFloat(match[3]), cp1Y: parseFloat(match[4]),
          cp2X: parseFloat(match[5]), cp2Y: parseFloat(match[6]),
          endX: parseFloat(match[7]), endY: parseFloat(match[8]),
          stroke: p.getAttribute("stroke"),
          strokeWidth: p.getAttribute("stroke-width"),
          opacity: p.getAttribute("opacity")
        });
      }
    });
    setCurves(parsedCurves);
  }, [svgString]);

  const handlePointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!draggingNode) return;
    const svg = e.currentTarget;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const svgP = pt.matrixTransform(ctm.inverse());

    setCurves(prev => prev.map(c => {
      if (c.id !== draggingNode.curveId) return c;
      const updated = { ...c };
      if (draggingNode.nodeType === 'start') { updated.startX = svgP.x; updated.startY = svgP.y; }
      else if (draggingNode.nodeType === 'cp1') { updated.cp1X = svgP.x; updated.cp1Y = svgP.y; }
      else if (draggingNode.nodeType === 'cp2') { updated.cp2X = svgP.x; updated.cp2Y = svgP.y; }
      else if (draggingNode.nodeType === 'end') { updated.endX = svgP.x; updated.endY = svgP.y; }
      return updated;
    }));
  };

  const handlePointerUp = () => {
    setDraggingNode(null);
  };

  const downloadEditedSVG = () => {
    let svgContent = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 511 512" width="511" height="512">\n`;
    svgContent += `  <rect width="511" height="512" fill="rgb(238,236,235)" />\n`;
    curves.forEach(c => {
      const d = `M ${c.startX.toFixed(1)} ${c.startY.toFixed(1)} C ${c.cp1X.toFixed(1)} ${c.cp1Y.toFixed(1)} ${c.cp2X.toFixed(1)} ${c.cp2Y.toFixed(1)} ${c.endX.toFixed(1)} ${c.endY.toFixed(1)}`;
      svgContent += `  <path d="${d}" stroke="${c.stroke}" stroke-width="${c.strokeWidth}" fill="none" opacity="${c.opacity}" stroke-linecap="round" />\n`;
    });
    svgContent += `</svg>`;

    const blob = new Blob([svgContent], {type: 'image/svg+xml'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'edited_artwork.svg';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{width: '100%', height: '100%', position: 'relative'}}>
      <svg 
        viewBox="0 0 511 512" 
        style={{width: '100%', height: '100%', cursor: draggingNode ? 'grabbing' : 'default'}}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
      >
        {/* Render actual paths */}
        {curves.map(c => (
          <g key={c.id}>
            {/* Invisible thick path for easier clicking */}
            <path 
              d={`M ${c.startX} ${c.startY} C ${c.cp1X} ${c.cp1Y} ${c.cp2X} ${c.cp2Y} ${c.endX} ${c.endY}`} 
              stroke="transparent" 
              strokeWidth="15" 
              fill="none" 
              style={{cursor: 'pointer'}}
              onPointerDown={() => setSelectedCurve(c.id)}
            />
            {/* Actual visible path */}
            <path 
              d={`M ${c.startX} ${c.startY} C ${c.cp1X} ${c.cp1Y} ${c.cp2X} ${c.cp2Y} ${c.endX} ${c.endY}`} 
              stroke={c.stroke || 'black'} 
              strokeWidth={c.strokeWidth || '1'} 
              fill="none" 
              opacity={c.opacity || '1'} 
              style={{pointerEvents: 'none'}}
            />
          </g>
        ))}

        {/* Render control points for selected curve */}
        {selectedCurve !== null && curves.find(c => c.id === selectedCurve) && (() => {
          const c = curves.find(c => c.id === selectedCurve)!;
          return (
            <g>
              {/* Handles */}
              <line x1={c.startX} y1={c.startY} x2={c.cp1X} y2={c.cp1Y} stroke="#0000FF" strokeWidth="1" strokeDasharray="4" />
              <line x1={c.endX} y1={c.endY} x2={c.cp2X} y2={c.cp2Y} stroke="#0000FF" strokeWidth="1" strokeDasharray="4" />
              
              {/* Nodes */}
              {[
                { type: 'start', x: c.startX, y: c.startY },
                { type: 'cp1', x: c.cp1X, y: c.cp1Y },
                { type: 'cp2', x: c.cp2X, y: c.cp2Y },
                { type: 'end', x: c.endX, y: c.endY }
              ].map(node => (
                <circle 
                  key={node.type}
                  cx={node.x} cy={node.y} r="5" 
                  fill={node.type.startsWith('cp') ? '#0000FF' : '#FF00FF'} 
                  stroke="#000000" strokeWidth="1"
                  style={{cursor: 'grab'}}
                  onPointerDown={(e) => {
                    e.stopPropagation();
                    setDraggingNode({curveId: c.id, nodeType: node.type});
                  }}
                />
              ))}
            </g>
          );
        })()}
      </svg>
      <button 
        onClick={downloadEditedSVG}
        style={{
          position: 'absolute', bottom: '20px', left: '20px', 
          background: 'var(--color-bg-navy)', border: '1px solid var(--color-border-grid)',
          boxShadow: '3px 3px 0px var(--color-border-grid)',
          color: 'var(--color-text-primary)', padding: '8px 16px', borderRadius: '0',
          fontFamily: 'var(--font-display)', fontWeight: 600, fontSize: '14px', cursor: 'pointer',
          textTransform: 'uppercase'
        }}
      >
        ⭳ Download Edited SVG
      </button>
    </div>
  );
};
