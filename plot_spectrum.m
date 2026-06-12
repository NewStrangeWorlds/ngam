clearvars;

clearvars;

% Load the data
spectrum = ncread('output_terrestrial.nc', 'spectrum');
wavelength = ncread('output_terrestrial.nc', 'wavelength');


figure;

semilogx(wavelength, spectrum);

set(gca, 'FontSize', 13);
set(gca,'TickLabelInterpreter','latex');


xlabel("Wavelength ($\mu$m)",'Interpreter','latex');
ylabel("Flux (W m$^{-2}$ $\mu$m$^{-1}$)",'Interpreter','latex');


figure;

plot(wavelength, spectrum);

xlim([2 20]);

set(gca, 'FontSize', 13);
set(gca,'TickLabelInterpreter','latex');


xlabel("Wavelength ($\mu$m)",'Interpreter','latex');
ylabel("Flux (W m$^{-2}$ $\mu$m$^{-1}$)",'Interpreter','latex');