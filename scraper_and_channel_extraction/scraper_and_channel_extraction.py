from bs4 import BeautifulSoup
import pandas as pd
from pydub import AudioSegment
import traceback
def scraper(eaf_file):
    channel = 0
    init_path = 'Pueble-Nahuatl-Manifest/ELAN-files-Final-proofed-and-most-translated/'
    with open(init_path + eaf_file, 'r',encoding='utf-8') as f:
        data = f.read()

    soup = BeautifulSoup(data, "xml")

    # Find all <ANNOTATION> tags
    annotation_tags = soup.find_all('ANNOTATION')

    media_descriptor = soup.find('MEDIA_DESCRIPTOR')
    audio_file = media_descriptor['MEDIA_URL'].split('/')[-1]
    song = AudioSegment.from_wav('all_clips/'+ audio_file)
    
    temp = eaf_file.split('_')[2]
    var = soup.find('TIER')
    str_ = var['TIER_ID']
    str_=str_.strip()
    x = str_.split()
    check = ''
    
    for c in x:
        check+=c[0]
        
    if check == temp.split('-')[0][:3]:
        channel = 0
    else:
        channel = 1

    map_ = {
        'file':[],
        'transcription':[],
        'channel':[]
    }

    # Loop over the <ANNOTATION> tags
    for annotation_tag in annotation_tags:
        # Access elements within the <ANNOTATION> tag
        alignable_annotation = annotation_tag.find('ALIGNABLE_ANNOTATION')
        if alignable_annotation is not None:
            annotation_id = alignable_annotation['ANNOTATION_ID']
            time_slot_ref1 = alignable_annotation['TIME_SLOT_REF1']
            time_slot_ref2 = alignable_annotation['TIME_SLOT_REF2']
            annotation_value = alignable_annotation.find('ANNOTATION_VALUE').text

            bt1 = soup.find('TIME_SLOT',{'TIME_SLOT_ID':time_slot_ref1})
            bt2 = soup.find('TIME_SLOT',{'TIME_SLOT_ID':time_slot_ref2})

            ts1 = int(bt1.get('TIME_VALUE'))
            ts2 = int(bt2.get('TIME_VALUE'))

            sub = song[ts1:ts2]
            sliced_file = 'sliced_clips/'+audio_file+'_'+annotation_id+'.wav'
            sub.export(sliced_file,format='wav')

            map_['file'].append(sliced_file)
            map_['transcription'].append(annotation_value)
            map_['channel'].append(channel)
        
    return map_
    
    
import os
from tqdm import tqdm
folder_path = 'Pueble-Nahuatl-Manifest/ELAN-files-Final-proofed-and-most-translated/'
c = 0
# List all files within the folder
file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
datas = []
for f in tqdm(file_list):
    try:
        datas.append(scraper(f))
    except Exception as e:
        print(f)
        print('EXCEPTION: ',e)
        traceback.print_exc()


all_datas = {'files':[],'transcriptions':[],'channels':[]}
for d in datas:
    for fi in d['file']:
        all_datas['files'].append(fi)
        
    for t in d['transcription']:
        all_datas['transcriptions'].append(t)
        
    for c in d['channel']:
        all_datas['channels'].append(c)

import pandas as pd

df = pd.DataFrame(all_datas)

df.to_csv('all_data.csv', index=False)

import librosa
import soundfile as sf
from tqdm import tqdm

def extract_channel(audio_file,channel_index):
    y, sr = librosa.load(audio_file, sr=None, mono=False)

    selected_channel = y[channel_index]
    
    file = audio_file.split('/')[-1]
    output_file = 'channled_audio/'+ file
    sf.write(output_file, selected_channel, sr)

for i in tqdm(range(len(all_datas['files']))):
    extract_channel(all_datas['files'][i],all_datas['channels'][i])




