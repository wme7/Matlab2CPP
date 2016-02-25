function u0 = Set_IC_MultiDomain1d(tid,u0,SNX,OMP_THREADS)

for i = 2:SNX+1
    u0(i) = 0.0;
end

% if (tid==0)
%     for i = 2:SNX+1
%         u0(i) = 0.0;
%         % but ...
%         %if (i==2),  u0(i) = 0.0; end
%     end
% end
% if (tid>0 && tid<OMP_THREADS)
%     for i = 2:SNX+1
%         u0(i) = 0.0;
%     end
% end
% if (tid==OMP_THREADS-1)
%     for i = 2:SNX+1
%         u0(i) = 0.0;
%         % but ...
%         %if (i==SNX+1), u0(i) = 1.0; end
%     end
% end