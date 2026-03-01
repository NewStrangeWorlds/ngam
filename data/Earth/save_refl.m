clearvars;

data = readmatrix("global_surface_reflectance_0.3-25um.csv");

mu = data(:,2);
refl = data(:,4);

output(1,:) = mu;
output(2,:) = refl;

%dlmwrite('earth_spectral_surface_reflection.dat',output,'delimiter','\quad','precision','%.5e');

fileID = fopen('earth_spectral_surface_reflection.dat','w');
fprintf(fileID,'%1.4f  %1.4f\n',output);
fclose(fileID);

% 
% figure;
% 
% plot(mu, refl);