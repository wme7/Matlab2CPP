function u = Call_Update2d(tid,u,ut,SNY,SNX,NY,NX)

for j = 1:SNY;
    for i = 1:SNX;           
      u(i+1+(tid*SNX)+(NX+2)*j) = ut(i+1+(SNX+2)*j);
    end
end