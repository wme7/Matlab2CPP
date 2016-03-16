function myplot(result,nx,ny,nz)

L = 10;
W = 10;
H = 20;

[x,y,z] = meshgrid(linspace(0,L,nx),linspace(0,W,ny),linspace(0,H,nz));
V = reshape(result(:,4),nx,ny,nz);

h = slice(x,y,z,V,L/2,W/2,H/2);
h(1).EdgeColor = 'none';
h(2).EdgeColor = 'none';
h(3).EdgeColor = 'none';
% colormap hot;
axis equal;