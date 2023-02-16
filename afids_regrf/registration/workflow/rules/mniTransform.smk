def get_moving_vol(wc):
    if len(sessions)==0:
        if len(runs)>0:
            moving_volume = bids_name(root=join(config['input_dir'],'bids'),kind='anat',subject=wc.subject,session=None,suffix=suffix, **{'run':'01'})
        else:
            moving_volume = bids_name(root=join(config['input_dir'],'bids'),kind='anat',subject=wc.subject,session=None,suffix=suffix)
    else:
        if len(runs)>0:
            moving_volume = expand(bids_name(root=join(config['input_dir'],'bids'),kind='anat',subject=wc.subject,session='{session}',suffix=suffix, **{'run':'01'}),session=sessions, allow_missing=True)
        else:
            moving_volume = expand(bids_name(root=join(config['input_dir'],'bids'),kind='anat',subject=wc.subject,session='{session}',suffix=suffix),session=sessions, allow_missing=True)
    return moving_volume

rule align_mni_rigid:
    output:
        warped=bids_name(root=join(config['output_dir'],'derivatives','afids_mni'),kind='anat',subject='{subject}',suffix=suffix,**{'space':'MNI152NLin2009cAsym'}),
        xfm=bids_name(root=join(config['output_dir'],'derivatives','afids_mni'),kind='anat',subject='{subject}',suffix='xfm.mat',**{'space':'MNI152NLin2009cAsym'}),
    params:
        moving=get_moving_vol,
        fixed = config['template'],
        dof = config['flirt']['dof'],
        coarse = config['flirt']['coarsesearch'],
        fine = config['flirt']['finesearch'],
        cost = config['flirt']['cost'],
        interp = config['flirt']['interp'],
    envmodules: 'fsl'
    #log: 'logs/align_mni_rigid/sub-{subject}_T1w.log'
    shell:
        'flirt -in {params.moving} -ref {params.fixed} -out {output.warped} -omat {output.xfm} -dof {params.dof} -coarsesearch {params.coarse} -finesearch {params.fine} -cost {params.cost} -interp {params.interp}'

rule fsl_to_ras:
    input:
        xfm=bids_name(root=join(config['output_dir'],'derivatives','afids_mni'),kind='anat',subject='{subject}',suffix='xfm.mat',**{'space':'MNI152NLin2009cAsym'}),
        warped=bids_name(root=join(config['output_dir'],'derivatives','afids_mni'),kind='anat',subject='{subject}',suffix=suffix,**{'space':'MNI152NLin2009cAsym'}),
    params:
        moving_vol = get_moving_vol,
    output:
        xfm_new=bids_name(root=join(config['output_dir'],'derivatives','afids_mni'), subject='{subject}', suffix='xfm.txt', **{'space':'MNI152NLin2009cAsym','desc':'ras'}),
        tfm_new=bids_name(root=join(config['output_dir'],'derivatives','afids_mni'), subject='{subject}', suffix='xfm.tfm', **{'space':'MNI152NLin2009cAsym','desc':'ras'}),
    shell:
        'resources/c3d_affine_tool -ref {input.warped} -src {params.moving_vol} {input.xfm} -fsl2ras -o {output.xfm_new} && \
        resources/c3d_affine_tool -ref {input.warped} -src {params.moving_vol} {input.xfm} -fsl2ras -oitk {output.tfm_new}'

rule fid_tform_mni_rigid:
    input:
        xfm_new=bids_name(root=join(config['output_dir'],'derivatives','afids_mni'), subject='{subject}', suffix='xfm.txt',**{'space':'MNI152NLin2009cAsym','desc':'ras'}),
    params:
        bids_name(root=join(config['input_dir'], 'derivatives', 'afids_groundtruth'), subject='{subject}', session=None, suffix=suffix_afids, **{'space':'T1w','desc':'groundtruth'}),
        template = 'resources/dummy.fcsv',
    output:
        fcsv_new=bids_name(root=join(config['output_dir'],'derivatives','afids_mni'), subject='{subject}', suffix=suffix_afids, **{'space':'MNI152NLin2009cAsym','desc':'groundtruth'}),
        reg_done = touch(bids_name(root=join(config['output_dir'],'derivatives','afids_mni'), subject='{subject}', suffix='registration.done')),
    #log: 'logs/fid_tform_mni_rigid/sub-{subject}_T1w.log'
    script:
        '../scripts/tform_script.py'