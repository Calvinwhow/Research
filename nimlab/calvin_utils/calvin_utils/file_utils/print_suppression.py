import os, sys

class HiddenPrints:
    '''
    This is much safer because you can not forget to re-enable stdout, 
    which is especially critical when handling exceptions.
    Use case:
        with HiddenPrints():
            print("This will not be printed")

        print("This will be printed as before")
    '''
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout