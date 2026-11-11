import os, urllib.request, sys

def download_gutenberg(eids, dest_dir, timeout=10):
    os.makedirs(dest_dir, exist_ok=True)
    results = {}
    for eid in eids:
        url = f'https://www.gutenberg.org/cache/epub/{eid}/pg{eid}.txt'
        dest_path = os.path.join(dest_dir, f'{eid}.txt')
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                # read content as bytes
                data = resp.read()
                # write to file
                with open(dest_path, 'wb') as f:
                    f.write(data)
            if os.path.getsize(dest_path) == 0:
                results[eid] = f'Error: downloaded file is empty.'
            else:
                results[eid] = 'OK'
        except Exception as e:
            results[eid] = f'Error: {e}'
    return results

if __name__ == '__main__':
    ids = ["28054", "2554", "600", "8117"]
    dest = os.path.join(os.getcwd(), 'data', 'raw')
    res = download_gutenberg(ids, dest)
    for eid, status in res.items():
        print(f'{eid}: {status}')
    # verify all OK
    all_ok = all(v == 'OK' for v in res.values())
    sys.exit(0 if all_ok else 1)

