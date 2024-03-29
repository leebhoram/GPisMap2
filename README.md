# GPisMap_v2 

This repository contains source codes and demo files for our paper **Online
Continuous Mapping using Gaussian Process Implicit Surfaces (GPIS)**, which is
presented at [IEEE ICRA 2019](https://www.icra2019.org/).

The previous repo [GPisMAP(https://github.com/leebhoram/GPisMap.git) is not maintained but kept as legacy. 
This repo contains some updates including slightly enhanced speed and simple python interfaces.
 
## License

Licensed under [GNU General Public License version 3](https://www.gnu.org/licenses/gpl-3.0.html).

## Requirements: Software

1. [Eigen](http://eigen.tuxfamily.org/)

2. (Option 1) Build the source via MATLAB mex and run demo & visualization scripts on [MATLAB](https://www.mathworks.com/products/matlab.html)
    Recommended for best visualization. 

3. (Option 2) Alternatively, build the source via cmake and run demo & visualization scripts in Python (tested with version 3.7 - 3.9 and CMake 3.20+).
    Visualization is limited, but useful in case you have no Matlab license.

## Compiling and Running

1. Clone this repository
```
git clone https://github.com/leebhoram/GPisMap2.git
cd GPisMap2
```

### MATLAB Option 

2. cd to the mex directive in MATLAB
```
cd <GPisMap2>/matlab/mex
```

3. Compile the mex functions by executing the make script.
    * Setup mex 
    ```
    mex -setup
    ```
    * Run the make scripts
    ```
    make_GPisMap
    make_GPisMap3
    ```

4. Run the demo scripts

First, 
```
cd <GPisMap2>/matlab
```
Then,

 * For 2D
 ```
 run('demo_gpisMap.m')
 ```

* For 3D 
```
run('demo_gpisMap3.m')
```

5. Trouble shooting
    * If mex complains about not finding eigen, configure the eigen path appropriately
        in both `make_GPisMap.m` and `make_GPisMap3.m`

### Python Option

2. Build the source
```
mkdir build && cd build
cmake ..
make -j $(nproc)
```

3. Run the demo scripts 
```
cd <GPisMap2>/python
```
   * For 2D 
   ```
   python test.py
   ```
   * For 3D 
   ```
   python test3d.py 
   ```
4. Python required packages (To-do)

## Author

[Bhoram Lee](https://github.com/leebhoram) E-mail: <first_name>.<last_name>@gmail.com

## Misc.

Code has been developed and tested on Ubuntu 22.04 

## Citation
 
If you find GPisMap/GPisMap_v2 useful in your research, please consider citing:
```
  @article{<blee-icra19>,
      Author = {Bhoram Lee, Clark Zhang, Zonghao Huang, and Daniel D. Lee},
      Title = {Online Continuous Mapping using Gaussian Process Implicit Surfaces},
      Journal = {IEEE ICRA},
      Year = {2019}
   }    
```
