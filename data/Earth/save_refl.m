clearvars;

data = readmatrix("earth_spectral_surface_reflection.dat");

mu = data(:,1);
refl = data(:,2);


figure;

plot(mu, refl);