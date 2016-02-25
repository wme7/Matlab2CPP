function u0 = Set_IC_MultiDomain2d(tid,u0,SNY,SNX,OMP_THREADS)

if (tid==0)
    for j = 1:SNY
        for i = 1:SNX
            o=(i+1)+(SNX+2)*j; u0(o) = 0.25;
            % but ...
            if (i==1),   u0(o) = 0.0; end
            if (j==1),   u0(o) = 0.0; end
            if (j==SNY), u0(o) = 1.0; end
        end
    end
end
if (tid>0 && tid<OMP_THREADS)
    for j = 1:SNY
        for i = 1:SNX
            o=(i+1)+(SNX+2)*j; u0(o) = 0.50;
            % but ...
            if (j==1),   u0(o) = 0.0; end
            if (j==SNY), u0(o) = 1.0; end
        end
    end
end
if (tid==OMP_THREADS-1)
    for j = 1:SNY
        for i = 1:SNX
            o=(i+1)+(SNX+2)*j; u0(o) = 0.75;
            % but ...
            if (j==1),   u0(o) = 0.0; end
            if (i==SNX), u0(o) = 1.0; end
            if (j==SNY), u0(o) = 1.0; end
        end
    end
end