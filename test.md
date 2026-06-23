# Figure: Integer-only NPU/HTP execution model (black & white)

> Inline SVG — render được trong VS Code Markdown preview (Cmd+Shift+V).
> Dùng cho `\label{fig:npu_int_exec}`. Chỉ 2 màu: đen + trắng.

<svg viewBox="0 0 940 300" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica, Arial, sans-serif">
  <defs>
    <marker id="arrow" markerWidth="9" markerHeight="9" refX="7.5" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,3 L0,6 z" fill="#000"/>
    </marker>
  </defs>

  <!-- ===== HOST left (float) ===== -->
  <rect x="20" y="120" width="150" height="92" rx="6" fill="#fff" stroke="#000" stroke-width="1.4"/>
  <text x="95" y="150" text-anchor="middle" font-size="14" font-weight="bold" fill="#000">HOST (CPU)</text>
  <text x="95" y="172" text-anchor="middle" font-size="12.5" fill="#000">FP32 input</text>
  <text x="95" y="195" text-anchor="middle" font-size="11" font-style="italic" fill="#000">float I/O</text>

  <line x1="170" y1="166" x2="198" y2="166" stroke="#000" stroke-width="1.4" marker-end="url(#arrow)"/>

  <!-- ===== Quantize (solid black) ===== -->
  <rect x="200" y="141" width="72" height="50" rx="6" fill="#000" stroke="#000"/>
  <text x="236" y="163" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#fff">Quantize</text>
  <text x="236" y="180" text-anchor="middle" font-size="10.5" fill="#fff">float&#8594;int8</text>

  <!-- boundary 1 -->
  <line x1="294" y1="48" x2="294" y2="272" stroke="#000" stroke-width="1.1" stroke-dasharray="6 5"/>
  <text x="294" y="40" text-anchor="middle" font-size="10" fill="#000">host &#8596; accelerator boundary</text>

  <line x1="272" y1="166" x2="332" y2="166" stroke="#000" stroke-width="1.4" marker-end="url(#arrow)"/>

  <!-- ===== ACCELERATOR (thicker border) ===== -->
  <rect x="310" y="70" width="320" height="192" rx="8" fill="#fff" stroke="#000" stroke-width="2.4"/>
  <text x="470" y="98" text-anchor="middle" font-size="13.5" font-weight="bold" fill="#000">HEXAGON HTP / NPU</text>

  <rect x="334" y="118" width="272" height="56" rx="6" fill="#fff" stroke="#000" stroke-width="1.4"/>
  <text x="470" y="142" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#000">Integer MAC datapath</text>
  <text x="470" y="161" text-anchor="middle" font-size="11.5" fill="#000">INT8 &#183; W8A8 (fixed-point)</text>

  <rect x="334" y="196" width="272" height="40" rx="6" fill="#fff" stroke="#000" stroke-width="1.2" stroke-dasharray="4 3"/>
  <text x="470" y="214" text-anchor="middle" font-size="12" font-weight="bold" fill="#000">&#10007; No internal floating-point tensors</text>
  <text x="470" y="229" text-anchor="middle" font-size="10" fill="#000">every intermediate value stays integer</text>

  <line x1="608" y1="166" x2="658" y2="166" stroke="#000" stroke-width="1.4" marker-end="url(#arrow)"/>

  <!-- boundary 2 -->
  <line x1="646" y1="48" x2="646" y2="272" stroke="#000" stroke-width="1.1" stroke-dasharray="6 5"/>

  <!-- ===== Dequantize (solid black) ===== -->
  <rect x="660" y="141" width="72" height="50" rx="6" fill="#000" stroke="#000"/>
  <text x="696" y="163" text-anchor="middle" font-size="12" font-weight="bold" fill="#fff">Dequant.</text>
  <text x="696" y="180" text-anchor="middle" font-size="10.5" fill="#fff">int8&#8594;float</text>

  <line x1="732" y1="166" x2="758" y2="166" stroke="#000" stroke-width="1.4" marker-end="url(#arrow)"/>

  <!-- ===== HOST right (float) ===== -->
  <rect x="760" y="120" width="150" height="92" rx="6" fill="#fff" stroke="#000" stroke-width="1.4"/>
  <text x="835" y="150" text-anchor="middle" font-size="14" font-weight="bold" fill="#000">HOST (CPU)</text>
  <text x="835" y="172" text-anchor="middle" font-size="12.5" fill="#000">FP32 output</text>
  <text x="835" y="195" text-anchor="middle" font-size="11" font-style="italic" fill="#000">float I/O</text>
</svg>


