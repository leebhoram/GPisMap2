%
% GPisMap - Online Continuous Mapping using Gaussian Process Implicit Surfaces
% https://github.com/leebhoram/GPisMap
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License v3 as published by
% the Free Software Foundation.
%
% This program is distributed in the hope that it will be useful, but WITHOUT
% ANY WARRANTY; without even the implied warranty of any FITNESS FOR A
% PARTICULAR PURPOSE. See the GNU General Public License v3 for more details.
%
% You should have received a copy of the GNU General Public License v3
% along with this program; if not, you can access it online at
% http://www.gnu.org/licenses/gpl-3.0.html.
%
% Authors: Bhoram Lee <bhoram.lee@gmail.com>
%

close all
clearvars

addpath('./plot_scripts');
addpath('./mex');
mexGPisMap3('reset')

% The original dataset downloadable at http://rll.berkeley.edu/bigbird/
% The following data are sampled and prepared by the authors for the test
depthpath = '../data/3D/bigbird_detergent/masked_depth';
poses = load('../data/3D/bigbird_detergent/pose/poses.txt');

% input sequence
FrameNums = poses(:,1);  
CamIDs = poses(:,2);
poses = single(poses(:,3:end));

% test 3D volume grid
[xg, yg, zg ] = meshgrid(-0.07:0.01:0.13, -0.1:0.01:0.14, 0:0.01:0.28);
xtest1 = single([xg(:)'; yg(:)'; zg(:)']);

for k=1:numel(FrameNums)
    frmNo = FrameNums(k);
    camID = CamIDs(k);

    D = imread(fullfile(depthpath,sprintf('frame%d_cam%d.png',frmNo,camID)));
    D = single(D)*single(0.0001); % 10 mm to meter

    T = reshape(poses(k,:),4,4);
    R = T(1:3,1:3);
    t = T(4,1:3)';

    mexGPisMap3('setCamera',camID,'bigbird'); % See mex/mexGPisMap3.cpp for camera calibration info
    mexGPisMap3('update',D,[t' reshape(R,1,[])]);

    close all;
    if  1 % k == k_last
       visualize_gpisMap3
    end

    % paus if needed
    % disp('press a button to continue');
    % pause;

end

% clear resources
mexGPisMap3('reset')
