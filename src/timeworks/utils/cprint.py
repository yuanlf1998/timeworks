# -*- coding: utf-8 -*-
# Author: Linfeng
# Date: 2025-12-09
# conditional print

def cprint(content, verbose=True, *args, **kwargs):
    if verbose:
        print(content, *args, **kwargs)
