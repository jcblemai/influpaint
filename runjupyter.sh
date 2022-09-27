#!/bin/bash


XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)

#module load  gcc/9.1.0 julia/1.6.3 dotnet/3.1.100 cuda/11.4 matlab/2022a anaconda/2021.11

echo "Open a local terminal (on your laptop) and copy this string into it:"
echo ""
echo -e "\tssh -N -L $ipnport:$ipnip:$ipnport $(whoami)@longleaf.unc.edu"
echo ""
echo "In this local terminal enter your ONYEN password. Window will hang - that's ok."
echo ""
echo "Next copy the following url into your local web browser:"
echo ""
echo -e "\tlocalhost:$ipnport"
echo ""
echo "This will bring up jupyter in your local web browser. Log in using the jupyter notebook password you created."
echo ""
echo "When you are finished with your session and have logged out of jupyter in your local web brower be sure to return to this Longleaf terminal and type Ctrl-C (it might be necessary to do Ctrl-C repeatedly a few times). You should also do Ctrl-C back in your local terminal."

jupyter lab --no-browser --port=$ipnport --ip=$ipnip 

wait

