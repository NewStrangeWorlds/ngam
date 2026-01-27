clearvars;

data = readmatrix("spectrum.dat");


figure;

loglog(data(:,1), data(:,2));