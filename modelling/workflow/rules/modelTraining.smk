def get_files(wc):
    if config['registration']:
        nii_files = glob(join(config['output_dir'],'deriv','mni_space','*','anat','*_space-MNI152NLin2009cAsym_'+suffix))
        fcsv_files = glob(join(config['output_dir'],'deriv','mni_space','*','anat','*_space-MNI152NLin2009cAsym_desc-ras_'+suffix_afids))
    else:
        nii_files = glob(join(config['output_dir'],'bids','*','anat','*_'+suffix))
        fcsv_files = glob(join(config['output_dir'],'deriv','afids','*','anat','*_space-T1w_*'+suffix_afids))
    files=dict(zip(nii_files, fcsv_files))
    return files

rule featureExtract:
    params:
        afid_num='{afid_num}',
        model_params=config['model_params'],
        nii_files=get_files,
        train_level='{train_level}',
    output:
        out_features,
    script:
        "../scripts/data_store.py"

rule modelTrain:
    input:
        out_features,
    params:
        model_params = config['model_params'],
    threads: workflow.cores * config['model_params']['num_threads']
    resources:
        mem_mb=config['model_params']['max_memory']
    output:
        out_models,
    script:
        "../scripts/train.py"