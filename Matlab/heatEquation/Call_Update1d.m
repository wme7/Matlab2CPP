function u = Call_Update1d(tid,u,ut,SNX,NX)

for i=1:SNX; 
    u((i+1+tid*SNX))=ut(i+1); 
end
