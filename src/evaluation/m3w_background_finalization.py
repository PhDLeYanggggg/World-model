"""Forward dictionary-assembly repair; frozen matching and grouping stay intact."""


def summarize_roles_once(module,*args):
    original=module.conservative_exclusions
    seen=[]

    def without_duplicate(*values):
        grouping=original(*values)
        if grouping.get('scientific_roles_assigned') is not False:
            raise ValueError('Finalization cannot grant scientific roles')
        seen.append(grouping)
        return {key:value for key,value in grouping.items() if key!='scientific_roles_assigned'}

    module.conservative_exclusions=without_duplicate
    try:
        summary=module.summarize(*args)
    finally:
        module.conservative_exclusions=original
    if len(seen)!=1 or any(summary.get(key)!=value for key,value in seen[0].items()):
        raise ValueError('Finalization changed original grouping fields')
    return summary
