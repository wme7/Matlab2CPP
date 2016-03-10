function [u,ut] = Call_Comms1d(tid,u,ut,SNX,NX,OMP_THREADS)

u((  2  +tid*SNX))=ut(  2  );   
u((SNX+1+tid*SNX))=ut(SNX+1);

ut(  1  )=u(  1  +tid*SNX);     
ut(SNX+2)=u(SNX+2+tid*SNX);
