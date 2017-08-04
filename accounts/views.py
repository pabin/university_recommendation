from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from accounts.models import student_info
from accounts.models import users_university
from .models import User
from .forms import UserForm
from .forms import StudentForm
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import components
from django.shortcuts import render, redirect


# --system login function--
def login_user(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)

        if user is not None:
            if user.is_active:
                login(request, user)
                return redirect('/accounts/profile')

        else:
            return render(request, 'login.html', {'error_message': 'Username or Password doesn\'t match'})

    return render(request, 'login.html')


# --logout function--
def logout_user(request):
    logout(request)
    form = UserForm(request.POST or None)
    context = {
        "form": form,
    }
    return render(request, 'login.html', context)


# --Register New Accounts for New user--
def register(request):
    if request.method == 'POST':
        basic_forms = UserForm(request.POST)
        detail_forms = StudentForm(request.POST)

        if basic_forms.is_valid() and detail_forms.is_valid():
            user = basic_forms.save(commit=False)
            username = basic_forms.cleaned_data.get('username')
            password = basic_forms.cleaned_data.get('password')
            user.set_password(password)
            user.save()

            detail_form = detail_forms.save(commit=False)
            detail_form.user = user
            detail_form.save()

            user = authenticate(username=username, password=password)
            login(request, user)
            return redirect('/')
    else:
        basic_forms = UserForm()
        detail_forms = StudentForm()
    return render(request, 'register.html', {'basic_form': basic_forms, 'detail_form': detail_forms})


# --View User profiles--
def display_student_profile(request):
    user = User.objects.get(username=request.user.username)
    all_student_info = student_info.objects.filter(user=user)

    for major in all_student_info:
        intended_major = major.Intended_Major
        print (intended_major)

    context = {
        'all_student_info': all_student_info}
    return render(request, 'profile.html', context)


def edit_profile(request):
    user = User.objects.get(username=request.user.username)
    all_student_info = student_info.objects.filter(user=user)

    for student_details in all_student_info:

        student_details_form = StudentForm(request.POST or None,
                                           initial={'Intended_Major': student_details.Intended_Major,
                                                    'UnderGrad_GPA': student_details.UnderGrad_GPA,
                                                    'GRE_Verbal_Score': student_details.GRE_Verbal_Score,
                                                    'GRE_Quant_Score': student_details.GRE_Quant_Score,
                                                    'GRE_AWA_Score': student_details.GRE_AWA_Score,
                                                    'TOEFL_Score': student_details.TOEFL_Score,
                                                    'Student_Status': student_details.Student_Status,
                                                    'Degree_applying': student_details.Degree_applying,
                                                    })

        if request.method == 'POST':
            student_details.Intended_Major = request.POST['Intended_Major']
            student_details.UnderGrad_GPA = request.POST['UnderGrad_GPA']
            student_details.GRE_Verbal_Score = request.POST['GRE_Verbal_Score']
            student_details.GRE_Quant_Score = request.POST['GRE_Quant_Score']
            student_details.GRE_AWA_Score = request.POST['GRE_AWA_Score']
            student_details.TOEFL_Score = request.POST['TOEFL_Score']
            student_details.Student_Status = request.POST['Student_Status']
            student_details.Degree_applying = request.POST['Degree_applying']

            student_details.save()

            return redirect('/accounts/profile')

        context = {
            "student_details_form": student_details_form
        }

    return render(request, "edit_profile.html", context)

# ------Add New Universities to User Dashboard and Displays universities------
def student_dashboard(request):
    if not request.user.is_authenticated():
        return render(request, 'login.html')
    else:
        query = request.GET.get('q')

        if query is not None:

            university = users_university(university_name=query, user=request.user)

            university.save()

            user = User.objects.get(username=request.user.username)
            universities_added = users_university.objects.filter(user=user)

            context = {
                'universities_added': universities_added}
            return render(request, 'dashboard.html', context)
        else:
            user = User.objects.get(username=request.user.username)
            universities_added = users_university.objects.filter(user=user)

            context = {
                'universities_added': universities_added}
            return render(request, 'dashboard.html', context)


def delete_dashboard_university(request, university_id):
    university = users_university.objects.get(pk=university_id)
    university.delete()
    remaining_university = users_university.objects.filter(user=request.user)
    return render(request, 'dashboard.html', {'universities_added': remaining_university})


def about_us(request):
    plot = figure(plot_width=600, plot_height=500, y_range=(0, 100))

    plot.vbar(x=[1], width=0.5, bottom=0, top=[85], color=["green"], legend="Pabin R Luitel",
              alpha=0.7)

    plot.vbar(x=[2], width=0.5, bottom=0, top=[10],
              color=["blue"], legend="Kshitiz Shrestha", alpha=0.7)

    plot.vbar(x=[3], width=0.5, bottom=0, top=[5],
              color=["red"], legend="Sabin Silwal", alpha=0.7)

    plot.yaxis.axis_label = "Percent"

    script, div = components(plot, CDN)

    context = {
        'script': script,
        'div': div
    }
    return render(request, 'about.html', context)

