function [f, g, h] = EngProb2_PlanetaryGearTrain(x)
if nargin == 0
    f = struct('n', 9, 'ng', 10, 'nh', 1, 'lb', [16.51,13.51,13.51,16.51,13.51,47.51,0.51,0.51,0.51], 'ub', [96.49,54.49,51.49,46.49,51.49,124.49,3.49,6.49,6.49]);
    g = []; h = [];
    return
end
ps = size(x,1);
        x = round(abs(x)); Pind = [3,4,5]; mind = [ 1.75, 2, 2.25, 2.5, 2.75, 3.0];
        N1 = x(:,1); N2 = x(:,2); N3 = x(:,3); N4 = x(:,4); N5 = x(:,5); N6 = x(:,6);
        p  = Pind(:,x(:,7))'; m1 = mind(:,x(:,8))';m2 = mind(:,x(:,9))';
        i1 = N6./N4; i01 = 3.11;
        i2 = N6.*(N1.*N3+N2.*N4)./(N1.*N3.*(N6-N4)); i02 = 1.84;
        iR = -(N2.*N6./(N1.*N3)); i0R = -3.11;
        f  = max([i1-i01,i2-i02,iR-i0R],[],2);
        Dmax = 220; dlt22 = 0.5; dlt33 = 0.5; dlt55 = 0.5; dlt35 = 0.5; dlt34 = 0.5; dlt56 = 0.5;
        beta = acos(((N6-N3).^2+(N4+N5).^2-(N3+N5).^2)./(2.*(N6-N3).*(N4+N5)));
        g(:,1) = m2.*(N6+2.5)-Dmax;
        g(:,2) = m1.*(N1+N2)+m1.*(N2+2)-Dmax;
        g(:,3) = m2.*(N4+N5)+m2.*(N5+2)-Dmax;
        g(:,4) = abs(m1.*(N1+N2)-m2.*(N6-N3))-m1-m2;
        g(:,5) = -((N1+N2).*sin(pi./p)-N2-2-dlt22);
        g(:,6) = -((N6-N3).*sin(pi./p)-N3-2-dlt33);
        g(:,7) = -((N4+N5).*sin(pi./p)-N5-2-dlt55);
        if beta == real(beta)
        g(:,8) = (N3+N5+2+dlt35).^2-((N6-N3).^2+(N4+N5).^2-2.*(N6-N3).*(N4+N5).*cos(2.*pi./p-beta));
        else
        g(:,8) = 1e6;
        end
        g(:,9) = -(N6-2.*N3-N4-4-2.*dlt34);
        g(:,10) = -(N6-N4-2.*N5-4-2.*dlt56);
        h(:,1)  = rem(N6-N4,p);
end
