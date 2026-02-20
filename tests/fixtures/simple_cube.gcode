; Simple calibration cube 20x20mm
G28 ; Home all axes
M104 S210 ; Set hotend temp
M140 S60 ; Set bed temp
M109 S210 ; Wait hotend
M190 S60 ; Wait bed
M106 S255 ; Fan on
G90 ; Absolute positioning
M82 ; Absolute extruder

; Layer 1
G1 Z0.2 F300
G1 X10 Y10 F3000 ; Move to start
G1 X30 Y10 E0.5 F1200 ; Front edge
G1 X30 Y30 E1.0 ; Right edge
G1 X10 Y30 E1.5 ; Back edge
G1 X10 Y10 E2.0 ; Left edge

; Layer 2
G1 Z0.4 F300
G1 X10 Y10 E2.5 F1200
G1 X30 Y10 E3.0
G1 X30 Y30 E3.5
G1 X10 Y30 E4.0
G1 X10 Y10 E4.5

; Layer 3
G1 Z0.6 F300
G1 X10 Y10 E5.0 F1200
G1 X30 Y10 E5.5
G1 X30 Y30 E6.0
G1 X10 Y30 E6.5
G1 X10 Y10 E7.0

M104 S0 ; Heater off
M140 S0 ; Bed off
M107 ; Fan off
G28 ; Home
