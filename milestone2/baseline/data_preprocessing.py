from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

def pd_to_jsonl(df, lang, train = True):
    df['id'] = [f'{lang}-sample' + str(i) for i in range(len(df))]
    if train:
        df_x = df[['id', f'{lang}_toxic_comment']]
        df_y = df[['id', f'{lang}_neutral_comment']]

        df_x = df[['id', f'{lang}_toxic_comment']].rename(columns={f'{lang}_toxic_comment': 'text'})
        df_y = df[['id', f'{lang}_neutral_comment']].rename(columns={f'{lang}_neutral_comment': 'text'})

        df_train_x, df_val_x, df_train_y, df_val_y = train_test_split(df_x, df_y, test_size=0.2, random_state=42)
        
        df_train_x.to_json(f'train/input_{lang}_training_dataset.jsonl', orient='records', lines=True, force_ascii=False)
        df_train_y.to_json(f'train/reference_{lang}_training_dataset.jsonl', orient='records', lines=True, force_ascii=False)
        df_val_x.to_json(f'dev/input_{lang}_validation_dataset.jsonl', orient='records', lines=True, force_ascii=False)
        df_val_y.to_json(f'dev/reference_{lang}_validation_dataset.jsonl', orient='records', lines=True, force_ascii=False)
    else:
        filename = f"test/{lang}_dev_dataset.jsonl"
        df = df.rename(columns = {'toxic_sentence' : 'text'})
        df.to_json(filename, orient='records', lines=True, force_ascii=False)


if __name__ == "__main__":
    # Loading training and dev dataset
    dataset_e = load_dataset("s-nlp/paradetox")
    dataset_r = load_dataset("s-nlp/ru_paradetox")
    dataset_dev = load_dataset("textdetox/multilingual_paradetox")

    # Data Preprocess for english + russian
    df_e = pd.DataFrame(dict(dataset_e)['train'])
    df_r = pd.concat([pd.DataFrame(dict(dataset_r)['train']), pd.DataFrame(dict(dataset_r)['validation'])])

    print('English Training Dataset Samples:')
    print(df_e.head())
    print('\nRussian Training Dataset Samples:')
    print(df_r.head())

    #export training pd to jsonl 
    pd_to_jsonl(df_e, 'en')
    pd_to_jsonl(df_r, "ru")

    # Automatically process each language subset for dev
    for lang, subset in dataset_dev.items():
        # Converting the subset to a Pandas DataFrame
        df_lang = pd.DataFrame(subset)  
        pd_to_jsonl(df_lang, lang, False)

