def normalize_genotype(geno):
    """
    Ensures genotype is uppercase and sorted alphabetically.
    Examples:
      'ga' → 'AG'
      'TG' → 'GT'
    """
    if not isinstance(geno, str):
        return None
    letters = sorted(geno.upper())
    return "".join(letters)


def most_common(lst):
    """
    Returns the most frequent item in a list.
    """
    if len(lst) == 0:
        return None
    return max(set(lst), key=lst.count)
    
