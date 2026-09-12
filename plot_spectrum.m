clearvars;

file_name = "output_gas.nc";

% Load the data
spectrum = ncread(file_name, 'spectrum');
wavelength = ncread(file_name, 'wavelength');


file_name = "output_gas_neovulcan.nc";

% Load the data
spectrum2 = ncread(file_name, 'spectrum');
wavelength2 = ncread(file_name, 'wavelength');


figure;

loglog(wavelength, spectrum, wavelength2, spectrum2);

set(gca, 'FontSize', 13);
set(gca,'TickLabelInterpreter','latex');

ylim([1e-2 inf]);


xlabel("Wavelength ($\mu$m)",'Interpreter','latex');
ylabel("Flux (W m$^{-2}$ $\mu$m$^{-1}$)",'Interpreter','latex');


figure;

plot(wavelength, spectrum);

xlim([2 20]);

set(gca, 'FontSize', 13);
set(gca,'TickLabelInterpreter','latex');


xlabel("Wavelength ($\mu$m)",'Interpreter','latex');
ylabel("Flux (W m$^{-2}$ $\mu$m$^{-1}$)",'Interpreter','latex');