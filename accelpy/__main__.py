"""main entry for accelpy command-line interface"""
  

def main():
    from accelpy.command import accelpy as app
    return app()

if __name__ == "__main__":
    sys.exit(main())
