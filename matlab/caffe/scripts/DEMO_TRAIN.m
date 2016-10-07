% demo_train.m
addpath(genpath('../util'));
addpath('../');
FILTER_TYPE = 'WMF';
TRAINI_PATH = '../../../../DATA/train2014'; % put ur own path

Solver = Filter_Train(FILTER_TYPE, TRAINI_PATH);
