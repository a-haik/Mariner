function MC_model = CSOV_MC_model(params)
%%
fns = ['data/SOV_05-Feb-2023.mat';'data/SOV_06-Feb-2023.mat';'data/SOV_07-Feb-2023.mat';
    'data/SOV_08-Feb-2023.mat';'data/SOV_09-Feb-2023.mat';'data/SOV_10-Feb-2023.mat'];
% fns = ['data/SOV_05-Feb-2023.mat'; 'data/SOV_06-Feb-2023.mat'];

t_c = cell(size(fns,1),1);
Pd_c = cell(size(fns,1),1);
t0 = 0;
for i = 1:size(fns,1)
    DL = load(fns(i,:));
    dat = DL.SOV_data;
    
    dat.date = datetime(dat.Timestamp_UTC_, 'InputFormat', 'yyy-MM-dd''T''HH:mm:ss.SSS');

    sumColumns = {'DG1_GeneratorActivePower','DG2_GeneratorActivePower',
        'DG3_GeneratorActivePower','DG4_GeneratorActivePower'};
    dat.powerDemand = sum(dat{:, sumColumns}, 2); % The second parameter '2' in sum() specifies that the summation is along rows
    dat.t = seconds(dat.date-dat.date(1));
    t = [0:dat.t(end)];
    t_c{i} = t+t0;
    t0 = t_c{i}(end);
    Pd_c{i} = interp1(dat.t, dat.powerDemand, t);
end
D.t = [t_c{:}];
D.Pd = [Pd_c{:}];


%%
Ds = downsample_block_mean(D, params.Ts);


%%
MC_model = fit_dtmc(Ds, params.n_states);
MC_model

%%
% % ZOH vectors (separate from downsampling)
% tL = Ds.left_edges; yb = Ds.Pd; T = Ds.Tsec;
% tzoh = reshape([tL.'; (tL+T).'], [], 1);
% yzoh = reshape([yb.'; yb.'], [], 1);
% 
% % Plot (ZOH)
% figure(1); 
% plot(D.t, D.Pd, tzoh, yzoh, 'LineWidth', 1.5);
% xlabel('Time [s]'); ylabel('Power [kW]');
% title(sprintf('Downsampled mean, ZOH (T = %d s)', Ds.Tsec));
% grid on;
% 
% figure(2);
% imagesc(MC_model.P,[0 1]); axis xy; colormap(parula); colorbar;
% xlabel('Next state'); ylabel('Current state');
% title(sprintf('P (\\Delta=%gs)',MC_model.Delta));
% 
% %%
% n = 120;                               % steps to simulate
% M = size(MC_model.P,1);
% c = cumsum(MC_model.pi/sum(MC_model.pi)); s = find(rand<=c,1);  % start ~ stationary
% S = zeros(n,1); S(1)=s;
% for k=2:n
%     c=cumsum(MC_model.P(S(k-1),:)); 
%     S(k)=find(rand<=c,1); 
% end
% X = MC_model.levels(S);                      % kW per state
% t = (0:n)*MC_model.Delta;                    % left/right edges
% figure(3);
% stairs(t/3600, [X, X(end)], 'LineWidth', 1.5); grid on
% xlabel('Time [h]'); ylabel('Power [kW]'); title('DTMC simulation (ZOH)')

end

%%
function MC = fit_dtmc(Dds, M)
% Fit a discrete-time Markov chain from fixed-step samples.
% Inputs:
%   Dds.t   : time vector (fixed step; used only for Delta)
%   Dds.Pd  : downsampled power (column or row)
%   M       : number of states (ignored if 'edges' is given)
%   alpha   : Dirichlet smoothing per row (default 0.5)
%
% Output:
%   MC.P     : transition matrix (MxM)
%   MC.N     : transition counts (MxM)
%   MC.pi    : stationary distribution (Mx1)
%   MC.edges : bin edges (Mx1+1)
%   MC.levels: state representatives (Mx1)
%   MC.Delta : fixed step (seconds)
%   MC.S     : state indices for each sample (Nx1)

    alpha = 0.5;
    x = Dds.Pd(:);
    if numel(x) < 3, error('Not enough samples.'); end

    % Discretize; drop any NaNs at open edges
    edges = make_edges_quantile(x, M);
    S = discretize(x, edges);
    mask = isfinite(S);
    S = S(mask);
    if numel(S) < 3, error('Too few valid binned samples.'); end

    % One-step counts
    N = accumarray([S(1:end-1), S(2:end)], 1, [M M], @sum, 0);

    % Row-wise Dirichlet smoothing and normalization
    P = N + alpha;
    rowsum = sum(P,2);
    zero_rows = rowsum == 0;
    if any(zero_rows)
        % If a row had no data and alpha==0, make it uniform
        P(zero_rows, :) = 1;
        rowsum = sum(P,2);
    end
    P = P ./ rowsum;

    % Stationary distribution: solve (P' - I)pi = 0, sum(pi)=1
    A = [P' - eye(M); ones(1,M)];
    b = [zeros(M,1); 1];
    pi = A\b;

    MC.P      = P;
    MC.N      = N;
    MC.pi     = pi;
    MC.edges  = edges;
    MC.levels = (edges(1:end-1)+edges(2:end))/2;
    MC.Delta  = Dds.Tsec;
    MC.S      = S;
end

function edges = make_edges_quantile(x, M)
% quantile-like edges without toolboxes
x = x(isfinite(x)); x = sort(x);
n = numel(x);
qs = linspace(0,1,M+1);
qs(1)=0; qs(end)=1;
edges = arrayfun(@(p) qsimple(x,p), qs);
% widen to include endpoints
edges(1)  = min(x(1), edges(1)) - eps(edges(1));
edges(end)= max(x(end),edges(end)) + eps(edges(end));
edges = unique(edges,'stable');      % just in case
if numel(edges) < M+1
    % fallback to equal-width
    edges = linspace(min(x), max(x), M+1);
end
end

function q = qsimple(x, p)
r = 1 + (numel(x)-1)*p;
i = floor(r); j = ceil(r);
if i==j, q = x(max(1,min(end,i))); return; end
w = r - i;
q = (1-w)*x(i) + w*x(j);
end
