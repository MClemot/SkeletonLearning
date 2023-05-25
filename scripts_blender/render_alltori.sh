#!/bin/bash

for i in '_noise0.0' '_noise0.003' '_noise0.005' '_noise0.01' '_noise0.03' '_trunc' '_trunc2' '_trunc3' '_trunc4' ; do
     /home/julie/softs/blender-3.4.1-linux-x64/blender --background --python torus.py --  $i 'Sine';
     /home/julie/softs/blender-3.4.1-linux-x64/blender --background --python torus.py --  $i 'coverage';
     /home/julie/softs/blender-3.4.1-linux-x64/blender --background --python torus.py -- $i 'l1';
done

for i in dtore_*.png; do
    convert $i -crop 1250x1200+410+230 -quality 100 $i ; 
done
     
