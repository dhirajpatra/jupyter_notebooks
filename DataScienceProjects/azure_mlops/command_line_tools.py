import click

@click.command()
@click.option('--count', default=1)
def hello(count):
    for x in range(count):
        click.echo('Hello World!')

if __name__ == '__main__':
    hello()