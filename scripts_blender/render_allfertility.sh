#!/bin/bash

for i in '0' '0.5' '1' '2' ; do
     #/home/julie/softs/blender-3.4.1-linux-x64/blender --background --python fertilitynoise.py --  $i 'coverage';
     #/home/julie/softs/blender-3.4.1-linux-x64/blender --background --python fertilitynoise.py --  $i 'Sine';
     #/home/julie/softs/blender-3.4.1-linux-x64/blender --background --python fertilitynoise.py --  $i 'siren';
     #/home/julie/softs/blender-3.4.1-linux-x64/blender --background --python fertilitynoise.py --  $i 'igr';
     /home/julie/softs/blender-3.4.1-linux-x64/blender --background --python fertilitynoise.py --  $i 'vc';
     #/home/julie/softs/blender-3.4.1-linux-x64/blender --background --python fertilitynoise.py --  $i 'mcs';
done

for i in fertility_v*.png; do
    convert $i -crop 1250x1200+410+230 -quality 100 ${i%%.png}_crop.png ; 
done
     
