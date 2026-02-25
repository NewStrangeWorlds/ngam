clearvars;

% Load the data
T   = ncread('output.nc', 'temperature');
P   = ncread('output.nc', 'pressure');
conv = ncread('output.nc', 'convective');
flux_div = ncread('output.nc', 'flux_total');

% Split into radiative and convective
rad  = conv == 0;
cnv  = conv == 1;

% figure; hold on
% plot(T(rad), P(rad), 'b.', 'MarkerSize', 12, 'DisplayName', 'Radiative')
% plot(T(cnv), P(cnv), 'r.', 'MarkerSize', 12, 'DisplayName', 'Convective')
% 
% set(gca, 'YScale', 'log', 'YDir', 'reverse')
% xlabel('Temperature [K]')
% ylabel('Pressure [bar]')
% legend('Location', 'best')

%If you want connected line segments coloured by zone instead of dots, you can loop over contiguous regions:


T    = ncread('output.nc', 'temperature');
P    = ncread('output.nc', 'pressure');
conv = ncread('output.nc', 'convective');

figure; hold on

% Find transitions
edges = [1; find(diff(conv) ~= 0) + 1; numel(conv) + 1];

for k = 1:numel(edges)-1
    idx = edges(k):edges(k+1)-1;
    % Overlap by one point so segments connect
    if edges(k+1)-1 < numel(conv)
        idx = [idx, edges(k+1)];
    end
    if conv(edges(k)) == 1
        plot(T(idx), P(idx), 'r-', 'LineWidth', 2)
    else
        plot(T(idx), P(idx), 'b-', 'LineWidth', 2)
    end
end

box on;

set(gca, 'YScale', 'log', 'YDir', 'reverse')
xlabel('Temperature (K)','Interpreter','latex')
ylabel('Pressure (bar)','Interpreter','latex')
legend({'Convective', 'Radiative'}, 'Location', 'northeast', 'Interpreter','latex')
legend('boxoff');
set(gca,'TickLabelInterpreter','latex');

ax=gca;
ax.FontSize = 13;


% figure;
% 
% loglog(abs(flux_div), P);
% set(gca, 'YScale', 'log', 'YDir', 'reverse')
% xlabel('Flux divergence')
% ylabel('Pressure [bar]')
% 
% ax=gca;
% ax.FontSize = 20;