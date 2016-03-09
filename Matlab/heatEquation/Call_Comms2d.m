function [u,ut] = Call_Comms2d(phase,tid,u,ut,SNY,SNX,NY,NX)

if phase==1, % thread --> host
    for j=1:SNY
        u(  2  +(tid*SNX)+(NX+2)*j) = ut(  2  +(SNX+2)*j);
        u(SNX+1+(tid*SNX)+(NX+2)*j) = ut(SNX+1+(SNX+2)*j);
    end
    for i=1:SNX
        u(i+1+(tid*SNX)+(NX+2)* 1 ) = ut(i+1+(SNX+2)* 1 );
        u(i+1+(tid*SNX)+(NX+2)*SNY) = ut(i+1+(SNX+2)*SNY);
    end
elseif phase==2, % thread <-- host
    for j=1:SNY
        ut(  1  +(SNX+2)*j) = u(  1  +(tid*SNX)+(NX+2)*j);
        ut(SNX+2+(SNX+2)*j) = u(SNX+2+(tid*SNX)+(NX+2)*j);
    end
    for i=1:SNX
        ut(i+1+(SNX+2)*   1   ) = u(i+1+(tid*SNX)+(NX+2)*   1   );
        ut(i+1+(SNX+2)*(SNY+1)) = u(i+1+(tid*SNX)+(NX+2)*(SNY+1));
    end
end
