from sqlalchemy import create_engine, text

engine = create_engine("sqlite:///face.db")

with engine.connect() as connection:
    result = connection.execute(text('select "Hello" '))

    print(result.all())