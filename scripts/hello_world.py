from cs336_basics.modal_utils import app, build_image


@app.function(image=build_image())
def hello(name: str):
    print(f"Hello, {name}!")


@app.local_entrypoint()
def modal_main():
    hello.remote("Modal")


if __name__ == "__main__":
    hello.local("Local")
