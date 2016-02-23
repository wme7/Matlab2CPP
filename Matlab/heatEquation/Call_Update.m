function u = Call_Update(tid,u,ut,SNY,SNX,NY,NX)

for j = 2:SNY-1;
    for i = 2:SNX-1;           
      u(((i-1)+tid*(SNX-2))+NX*(j-2)) = ut(i+SNX*(j-1));
    end
end
