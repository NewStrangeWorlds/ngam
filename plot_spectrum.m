clearvars;

data = readmatrix("spectrum.dat");
data2 = readmatrix("spectrum_old_disort.dat");


figure;

loglog(data(:,1), data(:,2), data2(:,1), data2(:,2));

set(gca, 'FontSize', 13);
set(gca,'TickLabelInterpreter','latex');


xlabel("Wavelength ($\mu$m)",'Interpreter','latex');
ylabel("Flux (W m$^{-2}$ cm)",'Interpreter','latex');