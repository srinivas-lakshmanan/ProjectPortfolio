
%top plot
figure()
plot(out.time,out.output_v);
title('Simulation Time vs Vehicle Velocity');
xlabel('Time');
ylabel('Vehicle Velocity');

%bottom plot
figure()
plot(out.time,out.light_distance);
title('Simulation Time vs Distance to light');
xlabel('Time');
ylabel('Distajce to Light');