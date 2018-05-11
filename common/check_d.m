function proceed = check_d(D)
rounded_D = round(D);
proceed = true;
if sum(sum(rounded_D ~= D)) > 0
    proceed = false;
end

if size(D,1) ~= size(D,2)
    proceed = false;
end