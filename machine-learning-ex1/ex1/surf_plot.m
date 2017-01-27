figure;

XX = data(:,1);
ZZ = data(:,2);
YY = data(:,3);

plot3(XX, ZZ, YY, 'rx')
grid on;
xlabel('size'); ylabel('price'); zlabel('bedrooms');

surf(data(:,1), data(:,2), y)
xlabel('size'); ylabel('bedrooms');

MM = zeros(4500, 7000);
for i=1:length(XX)
	MM(XX(i), floor(ZZ(i)/100)) = YY(i);
end

imagesc(MM)
colorbar;
colormap ("default");
grid on;


stem3(XX,ZZ,YY)