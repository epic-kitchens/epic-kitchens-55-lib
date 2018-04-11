from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
from epic_kitchens.data import splits


annotations_root_url = 'https://raw.githubusercontent.com/epic-kitchens/annotations/master/'
videos_root_url = 'https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/'

labels = HTTP.remote(annotations_root_url + 'EPIC_train_action_narrations.csv', keep_local=True)

def video(wildcards):
    video_id = wildcards.video_id
    participant_id = wildcards.participant_id

    video_path = "{}/{}".format(participant_id, video_id)
    if video_id in spilts['train']:
        url = videos_root_url + 'train/' + video_path
    elif:
        url = videos_root_url + 'test/' + video_path
    else:
        raise ValueError(
                "Could not find split for video '{}' from participant '{}'".\
                    format(video_id, participant_id))
    return HTTP.remote(url, keep_local=True)


rule frames_for_video:
    input: 
    output: "data/interim/frames/{participant_id}/{video_id}.tar"
    params:
        qscale=4,
        workdir=os.path.join(config['local_data_root'], '{video_id}-frames')
    singularity: "shub://dl-container-registry/ffmpeg"
    resources:
        gpu=1
    threads: MAX_THREADS
    shell:
        """
        set -ex
        trap "echo 'Cleaning up'; rm -rf {params.workdir}; exit 255" SIGINT SIGTERM
        echo $CUDA_VISIBLE_DEVICES
        nvidia-smi
        mkdir -p {params.workdir}
        mkdir -p $(dirname {output})
        ffmpeg \
               -hwaccel cuvid \
               -c:v h264_cuvid \
               -i {input} \
               -vf 'scale_npp=-2:{config[frame_height]},hwdownload,format=nv12' \
               -q:v {params.qscale} \
               -r {config[fps]} \
               {params.workdir}/{config[frame_format]}
        tar -cvf {output} -C {params.workdir} .
        rm -rf {params.workdir}
        """


rule frame_segments:
    input:
        frames="data/interim/frames/{participant_id}/{video_id}.tar",
        labels=labels
    output: "data/interim/frame-segments/{participant_id}/{video_id}.tar"
    params:
        workdir=os.path.join(config['local_data_root'], '{video_id}-segments')
    threads: 1
    shell:
        """
          trap "echo 'Cleaning up'; rm -rf {params.workdir}; exit 255" SIGINT SIGTERM
          if [[ -d {params.workdir} ]]; then
            rm -rf {params.workdir}
          fi
          mkdir -p {params.workdir}/all_frames {params.workdir}/segmented
          tar -xvf {input.frames} -C {params.workdir}/all_frames 2>&1 > /dev/null
          python -m epic.scripts.data_prep.split_frames_by_label \
                 --frame-format {config[frame_format]} \
                 --modality RGB \
                 --fps {config[fps]} \
                 {input.labels} \
                 {wildcards.video} \
                 {params.workdir}/all_frames \
                 {params.workdir}/segmented
          tar -cvhf {output} -C {params.workdir}/segmented . 2>&1 > /dev/null
        """


rule optical_flow_for_video:
    input: "data/interim/frames/{participant_id}/{video_id}.tar"
    output: protected("data/interim/flow_stride={stride}_dilation={dilation}_bound={bound}/{participant_id}/{video_id}.tar")
    params:
        workdir=os.path.join(config['local_data_root'], "{video_id}-frames-flow"),
    resources:
        gpu=1
    singularity: "shub://dl-container-registry/furnari-flow"
    threads: 2
    shell:
        # CUDA_VISIBLE_DEVICES masks the GPUs enumerated by the CUDA runtime
        # so we don't need to specify the GPU to use, we just pick the GPU
        # with device ID=0 which CUDA_VISIBLE_DEVICES ensures to be one allocated
        # to us.
        """
        set -ex
        trap "echo 'Cleaning up'; rm -rf {params.workdir} ; exit 255" SIGINT SIGTERM
        echo "CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES"
        nvidia-smi
        mkdir -p {params.workdir}/flow/{{x,y}}
        echo "Extracting frames from {input} to {params.workdir}"
        tar -xvf {input} -C {params.workdir} > /dev/null
        echo "Extraction complete"
        echo "Computing flow for {wildcards.video}"
        compute_flow \
                {params.workdir} \
                {config[frame_format]} \
                flow/%s/{config[frame_format]} \
                -g 0 \
                -d {wildcards.dilation} \
                -s {wildcards.stride} \
                -b {wildcards.bound}
        echo "Flow computation for {wildcards.video} complete"
        mv {params.workdir}/flow/x {params.workdir}/flow/u
        mv {params.workdir}/flow/y {params.workdir}/flow/v
        echo "Archiving flow to {output}"
        tar -cvf {output} -C {params.workdir}/flow . > /dev/null
        """


rule optical_flow_segments:
    output:
        ("data/interim/flow-segments_stride={stride}_dilation={dilation}_bound={bound}"
         "/{participant_id}/{video_id}.tar")
    input:
        flow="data/interim/flow_stride={stride}_dilation={dilation}_bound={bound}/{participant_id}/{video_id}.tar",
        labels=labels
    params:
        workdir=os.path.join(config['local_data_root'], '{video_id}-flow-segments')
    threads: 1
    shell:
        """
          trap "echo 'Cleaning up'; rm -rf {params.workdir}; exit 255" SIGINT SIGTERM
          if [[ -d {params.workdir} ]]; then
            rm -rf {params.workdir}
          fi
          mkdir -p {params.workdir}/flow {params.workdir}/segmented
          tar -xvf {input.flow} -C {params.workdir}/flow 2>&1 > /dev/null
          python -m epic.scripts.data_prep.split_frames_by_label \
                 --frame-format {config[frame_format]} \
                 --of-dilation {wildcards.dilation} \
                 --of-stride {wildcards.stride} \
                 --modality Flow \
                 --fps {config[fps]} \
                 {input.labels} \
                 {wildcards.video} \
                 {params.workdir}/flow \
                 {params.workdir}/segmented
          tar -cvhf {output} -C {params.workdir}/segmented . 2>&1 > /dev/null
        """
