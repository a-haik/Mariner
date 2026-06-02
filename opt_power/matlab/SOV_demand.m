function Ds = SOV_demand(params, fn)

DL = load(fn);
dat = DL.SOV_data;
dat.date = datetime(dat.Timestamp_UTC_, 'InputFormat', 'yyy-MM-dd''T''HH:mm:ss.SSS');

sumColumns = {'DG1_GeneratorActivePower','DG2_GeneratorActivePower',
    'DG3_GeneratorActivePower','DG4_GeneratorActivePower'};
dat.powerDemand = sum(dat{:, sumColumns}, 2); % The second parameter '2' in sum() specifies that the summation is along rows
dat.t = seconds(dat.date-dat.date(1));
t = [0:dat.t(end)];
Pd_full = interp1(dat.t, dat.powerDemand, t);

%%
D.t = t;
D.Pd = Pd_full;

%%
Ds = downsample_block_mean(D, params.Ts);
