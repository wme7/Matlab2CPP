function u0 = Set_IC_MultiDomain2d(tid,u0,SNY,SNX)

for j = 1:SNY+2
    for i = 1:SNX+2
        o=(i)+(SNX+2)*(j-1);
        %
        if (i>1 && i<SNX+2 && j>1 && j<SNY+2)
            switch tid
                case 0; u0(o) = 0.10;
                case 1; u0(o) = 0.25;
                case 2; u0(o) = 0.40;
                case 3; u0(o) = 0.50;
                case 4; u0(o) = 0.75;
                case 5; u0(o) = 0.90;
            end
        end
        %
        %u0(o)=0.0;
        %
    end
end

% if (tid==0)
%     for j = 1:SNY
%         for i = 1:SNX
%             o=(i+1)+(SNX+2)*j; u0(o) = 0.25;
%             % but ...
%             if (i==1),   u0(o) = 0.0; end
%             if (j==1),   u0(o) = 0.0; end
%             if (j==SNY), u0(o) = 1.0; end
%         end
%     end
% end
% if (tid>0 && tid<OMP_THREADS)
%     for j = 1:SNY
%         for i = 1:SNX
%             o=(i+1)+(SNX+2)*j; u0(o) = 0.50;
%             % but ...
%             if (j==1),   u0(o) = 0.0; end
%             if (j==SNY), u0(o) = 1.0; end
%         end
%     end
% end
% if (tid==OMP_THREADS-1)
%     for j = 1:SNY
%         for i = 1:SNX
%             o=(i+1)+(SNX+2)*j; u0(o) = 0.75;
%             % but ...
%             if (j==1),   u0(o) = 0.0; end
%             if (i==SNX), u0(o) = 1.0; end
%             if (j==SNY), u0(o) = 1.0; end
%         end
%     end
% end