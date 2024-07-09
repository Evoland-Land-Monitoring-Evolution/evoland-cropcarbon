from pathlib import Path

if __name__ == "__main__":
    CGLOPS_QML_TXT = r"/data/sigma/Evoland_GPP/prototype/V01/test_region/CGLOPS_GDMP300_color_table.clr"
    # open the file and read the content
    with open(CGLOPS_QML_TXT, 'r') as f:
        content = f.read()
        
    # Physical range
    PHY_RANGE = [0, 35000]
    # get the color table
    color_table = content.split('\n')
    lst_new_table = []
    # count the amount of colors in the color table
    # -1 for nodata value in the end
    color_count = len(color_table) - 1
    # loop through the color table and get the color values
    for i in range(0, len(color_table)):
        classvalue = color_table[i].split(' ')[-1]
        if not classvalue == 255:
            new_class = int((int(classvalue) / color_count) * (PHY_RANGE[1] - PHY_RANGE[0]) + PHY_RANGE[0]) # NOQA
        else:
            new_class = 65535
        # replace the first and last value with the new class value
        color_table_line = f'{new_class} {color_table[i].split(" ")[1]} {color_table[i].split(" ")[2]} {color_table[i].split(" ")[3]} {color_table[i].split(" ")[4]} {new_class}' # NOQA
        lst_new_table.append(color_table_line)
        
    # write the new color table to the file
    # define output name of the file
    out_name = 'GPP_10M_color_table.txt'
    outfold = Path(CGLOPS_QML_TXT).parent
    with open(Path(outfold).joinpath(out_name), 'w') as f:
        f.write('\n'.join(lst_new_table))
    