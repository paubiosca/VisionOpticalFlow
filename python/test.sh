#!/bin/bash

python horn.py -d "mysine" --alpha-min=0.001 --alpha-max=0.5 --alpha-step=0.005 --iterations=1000
python horn.py -d "nasa" --alpha-min=0.1 --alpha-max=500 --alpha-step=5 --iterations=1000
python horn.py -d "rubberwhale" --alpha-min=0.0001 --alpha-max=0.1 --alpha-step=0.01 --iterations=1000
python horn.py -d "rubic" --alpha-min=0.05 --alpha-max=0.8 --alpha-step=0.005 --iterations=1000 
python horn.py -d "square" --alpha-min=0.0001 --alpha-max=0.05 --alpha-step=0.0005 --iterations=1000
python horn.py -d "taxi" --alpha-min=0.05 --alpha-max=0.8 --alpha-step=0.005 --iterations=1000
python horn.py -d "yosemite" --alpha-min=0.1 --alpha-max=200 --alpha-step=2 --iterations=1000


python lucas.py -d "mysine" --window-min=3 --window-max=100 --window-step=2
python lucas.py -d "nasa" --window-min=3 --window-max=100 --window-step=10 
python lucas.py -d "rubberwhale" --window-min=3 --window-max=100 --window-step=2 
python lucas.py -d "rubic" --window-min=3 --window-max=100 --window-step=2  
python lucas.py -d "square" --window-min=3 --window-max=100 --window-step=2 
python lucas.py -d "taxi" --window-min=3 --window-max=100 --window-step=2 
python lucas.py -d "yosemite" --window-min=3 --window-max=100 --window-step=2 


python lucas.py -d "mysine" --window-min=3 --window-max=100 --window-step=2 --gaussian-kernel=True
python lucas.py -d "nasa" --window-min=3 --window-max=100 --window-step=2 --gaussian-kernel=True
python lucas.py -d "rubberwhale" --window-min=3 --window-max=100 --window-step=2 --gaussian-kernel=True
python lucas.py -d "rubic" --window-min=3 --window-max=100 --window-step=2  --gaussian-kernel=True
python lucas.py -d "square" --window-min=3 --window-max=100 --window-step=2 --gaussian-kernel=True
python lucas.py -d "taxi" --window-min=3 --window-max=100 --window-step=2 --gaussian-kernel=True
python lucas.py -d "yosemite" --window-min=3 --window-max=100 --window-step=2 --gaussian-kernel=True


python horn.py -d "nasa" --alpha-min=0.1 --alpha-max=1 --alpha-step=2 --iterations=100
python horn.py -d "nasa" --alpha-min=0.5 --alpha-max=1 --alpha-step=2 --iterations=100
python horn.py -d "nasa" --alpha-min=1.5 --alpha-max=1 --alpha-step=2 --iterations=100
python horn.py -d "nasa" --alpha-min=4.0 --alpha-max=1 --alpha-step=2 --iterations=100

python horn.py -d "nasa" --alpha-min=0.1 --alpha-max=1 --alpha-step=2 --iterations=1000
python horn.py -d "nasa" --alpha-min=0.5 --alpha-max=1 --alpha-step=2 --iterations=1000
python horn.py -d "nasa" --alpha-min=1.5 --alpha-max=1 --alpha-step=2 --iterations=1000
python horn.py -d "nasa" --alpha-min=4.0 --alpha-max=1 --alpha-step=2 --iterations=1000


python horn.py -d "rubic" --alpha-min=0.1 --alpha-max=1 --alpha-step=2 --iterations=100
python horn.py -d "rubic" --alpha-min=0.5 --alpha-max=1 --alpha-step=2 --iterations=100
python horn.py -d "rubic" --alpha-min=1.5 --alpha-max=1 --alpha-step=2 --iterations=100
python horn.py -d "rubic" --alpha-min=4.0 --alpha-max=1 --alpha-step=2 --iterations=100

python horn.py -d "rubic" --alpha-min=0.1 --alpha-max=1 --alpha-step=2 --iterations=1000
python horn.py -d "rubic" --alpha-min=0.5 --alpha-max=1 --alpha-step=2 --iterations=1000
python horn.py -d "rubic" --alpha-min=1.5 --alpha-max=1 --alpha-step=2 --iterations=1000
python horn.py -d "rubic" --alpha-min=4.0 --alpha-max=1 --alpha-step=2 --iterations=1000


python horn.py -d "taxi" --alpha-min=0.1 --alpha-max=1 --alpha-step=2 --iterations=1000

python horn.py -d "taxi" --alpha-min=0.1 --alpha-max=1 --alpha-step=2 --iterations=100
python horn.py -d "taxi" --alpha-min=0.5 --alpha-max=1 --alpha-step=2 --iterations=100
python horn.py -d "taxi" --alpha-min=1.5 --alpha-max=1 --alpha-step=2 --iterations=100
python horn.py -d "taxi" --alpha-min=4.0 --alpha-max=1 --alpha-step=2 --iterations=100

python horn.py -d "taxi" --alpha-min=0.1 --alpha-max=1 --alpha-step=2 --iterations=1000
python horn.py -d "taxi" --alpha-min=0.5 --alpha-max=1 --alpha-step=2 --iterations=1000
python horn.py -d "taxi" --alpha-min=1.5 --alpha-max=1 --alpha-step=2 --iterations=1000
python horn.py -d "taxi" --alpha-min=4.0 --alpha-max=1 --alpha-step=2 --iterations=1000
