function myplot(result,nx,ny,L,W)

x = linspace(-L/2,L/2,nx);
y = linspace(-W/2,W/2,ny);
u = reshape(result,nx,ny);

% Plots results
figure; imagesc(y,x,u); colorbar;
xlabel('x'); ylabel('y'), axis equal; axis tight; title('SSP-RK3+WENO5 pressure field')
print('acoustics2d','-dpng')

% Export Result for Paraview
nz=1;

% Open the file.
fid = fopen('result.vtk', 'w');
if fid == -1
    error('Cannot open file for writing.');
end
fprintf(fid,'# vtk DataFile Version 2.0\n');
fprintf(fid,'Volume example\n');
fprintf(fid,'BINARY\n');
fprintf(fid,'DATASET STRUCTURED_POINTS\n');
fprintf(fid,'DIMENSIONS %d %d %d\n',nx,ny,nz);
fprintf(fid,'ASPECT_RATIO %d %d %d\n',1,1,1);
fprintf(fid,'ORIGIN %d %d %d\n',0,0,0);
fprintf(fid,'POINT_DATA %d\n',nx*ny*nz);
fprintf(fid,'SCALARS Pressure float 1\n');
fprintf(fid,'LOOKUP_TABLE default\n');
fwrite(fid,result,'float','ieee-be');