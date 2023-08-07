from flask import Flask,render_template,redirect,request
import caption

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def marks():
    if request.method == 'POST':
        f = request.files['userfile']
        path = './static/{}'.format(f.filename)
        f.save(path)

        caption_ = caption.caption_this_image(path)
        print("the caption is: ",caption_)

        result_dic = {
            'image':path,
            'caption':caption_
        }

    return render_template('index.html',your_result = result_dic)

if __name__=='__main__':
    app.run(debug=True)