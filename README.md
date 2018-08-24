# Use OCR to extract tables from pdf file
## Requirement:
`python3` and the following package:

`skimage`

`textract`

`pdf2image`

`baiduaip`

## Example:
```
import os
import glob
def pdf2csv(pdf_path, output_path_folder=None, table_keyword='接待时间'):
    pages = getPDFpages(pdf_path, keyword=table_keyword)
    if pages == []:
        return -1 
    ims = pdf2image(pdf_path, pages)
    i = 0
    d = {}
    for im_ in ims:
        im = im_[5:-5,5:-5]
        im2csv = Image2Csv_CL(im)
        im2csv.set_tables_dict()
        if im2csv.tables_p_dict == {}:
            continue
        else:
            for _, p in im2csv.tables_p_dict.items():
                img = im[p[0]:p[-1],:]
                im2csv_new = Image2Csv_CL(img)
                #im2csv_new.plot()
                #im2csv_new.write_to_csv(filename.split('.png')[0] + "-" + str(i) + '.csv')
                i += 1
                df = im2csv_new.image2df()
                d[i] = df

    for idx, df in d.items():
        s = df.astype('str').applymap(lambda cell: '接待时间' in cell).sum().sum()
        if s>=1:
            if not output_path_folder:
                output_path = pdf_path.replace('.pdf',str(idx)) + '.csv'
            else:
                output_path = os.path.join(output_path_folder, pdf_path.split('/')[-1].replace('.pdf', str(idx))) + '.csv'
            df.to_csv(output_path, index=None, header=False) 
            print(f'Success, write to file:{output_path}')
    return 0
#pdf2csv('/Users/sfi/Desktop/pdfs/000518 四环生物-2011.pdf', '/Users/sfi/Desktop')
```
