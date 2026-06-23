function Dds = downsample_block_mean(D, Tsec, align)
% Block-average power to period Tsec (seconds).
% align: 't0' (bins start at D.t(1)) or 'wall' (bins at multiples of Tsec).

    if nargin < 2, Tsec = 60; end
    if nargin < 3, align = 't0'; end

    t = D.t(:); y = D.Pd(:);
    ok = isfinite(t) & isfinite(y);
    t = t(ok); y = y(ok);
    [t, k] = sort(t); y = y(k);

    switch lower(align)
        case 't0'
            t0 = t(1);
        case 'wall'
            t0 = floor(t(1)/Tsec)*Tsec;  % align to nearest lower multiple of Tsec
        otherwise
            error('align must be ''t0'' or ''wall''.');
    end

    b = floor((t - t0)/Tsec);          % integer bin id
    bmin = b(1); bmax = b(end);
    nb = bmax - bmin + 1;
    b1 = b - bmin + 1;                 % 1..nb

    sumy = accumarray(b1, y, [nb 1], @sum, 0);
    cnt  = accumarray(b1, 1, [nb 1], @sum, 0);
    ybar = sumy ./ max(cnt,1);

    tL   = t0 + ((bmin:bmax)')*Tsec;   % left edges
    keep = cnt > 0;                    % drop truly empty bins

    Dds.t          = tL(keep) + Tsec/2;   % bin centers
    Dds.Pd         = ybar(keep);          % block means
    Dds.left_edges = tL(keep);
    Dds.Tsec       = Tsec;
    Dds.align      = align;
end
