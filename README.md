

## Useful repo cloned in repo:
- git clone https://github.com/cmu-delphi/delphi-epidata.git
- git clone https://github.com/jcblemai/Flusight-forecast-data.git
- git clone https://github.com/openai/guided-diffusion.git
- git clone https://github.com/andreas128/RePaint.git

then
```
cp delphi-epidata/src/client/delphi_epidata.py .
```



## Build conda enviroment
Build conda enviroment:
```bash
module purge
module load anaconda/2021.11.ood gcc/9.1.0 cuda/11.4 julia/1.6.3 matlab/2022a dotnet/3.1.100
module add anaconda ## change from the on demand conda... I think it's necessary to have the right to modify stuff...
# install jupyter on base environement:
conda activate base 
conda install ipykernel
conda install -c conda-forge ipywidgets

conda create -c conda-forge -n diffusion_torch2 seaborn scipy numpy pandas matplotlib ipykernel xarray netcdf4 h5netcdf tqdm
conda activate diffusion_torch2
conda install torchvision -c pytorch
python -m ipykernel install --user --name diffusion_torch2 --display-name "Python (diffusion_torch22)"  # typo, but kept
conda install -c conda-forge einops ipywidgets

```

## Run the diffusion
Create synthetic data padded from the data notebook
Run the diffusion from HF-annotated_diffusion.
### run on a Volta GPU at UNC

Create jupyter lab password (do once):
```bash
sh create_notebook_password.sh
```

Launch a job for 18h:
```bash
srun --ntasks=1 --cpus-per-task=4 --mem=32G --time=18:00:00 --partition=volta-gpu --gres=gpu:1 --qos=gpu_access --output=out.out sh runjupyter.sh &
```
then `cat out.out` which shows the instructions to go and make the ssh tunnel to connect on jupyter lab.

### Run on OpenOndemand
https://ondemand.rc.unc.edu (also very convenient to upload files/download figures)
Run a juypter notebook with request-gpu and
```
--mem=150gb -p volta-gpu --qos=gpu_access --gres=gpu:1
```
(I don't this is really necessary, because OOD GPU is not powerful enough (I think it’s an A100, but divided into small MIG 1g.5gb).
as Additional Job Submission Arguments and
```
"/nas/longleaf/home/chadi/inpainting-idforecasts"
```
as Jupyter startup directory