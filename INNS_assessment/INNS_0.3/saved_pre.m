none = [0.3944;0.3944;0.3750;0.3635;0.3485;0.3465;0.3521;0.3303;0.3438;0.3344];
meaned = [0.4109;0.3971;0.3756;0.3688;0.3582;0.3559;0.3506;0.3241;0.3291;0.3441];
map = [0.3788;0.4041;0.3579;0.3747;0.3674;0.3412;0.3585;0.3579;0.3388;0.3312];
meanmap = [0.3915;0.3918;0.3815;0.3579;0.3691;0.3494;0.3524;0.3435;0.3250;0.3288];
mapstded = [0.4;0.3865;0.3965;0.3929;0.3612;0.3865;0.3521;0.3474;0.3403;0.3397;0.3453];

hold off
plot(none,'r')
hold on
plot(meaned,'g')
plot(map,'b')
plot(meanmap,'c')
plot(mapstded, 'm')