from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from accounts.models import student_info
from .models import User
from accounts.forms import EditProfileForm
from .forms import UserForm


# --system logout function--
def login_user(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)

        if user is not None:
            if user.is_active:
                login(request, user)
                return render(request, 'profile.html')
            else:
                return render(request, 'login.html', {'error_message':
                                                      'Your account has been disabled'})
        else:
            return render(request, 'login.html', {'error_message': 'Invalid login'})
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
    form = UserForm(request.POST or None)
    if form.is_valid():
        user = form.save(commit=False)
        username = form.cleaned_data['username']
        password = form.cleaned_data['password']
        user.set_password(password)
        user.save()

        user = authenticate(username=username, password=password)
        if user is not None:
            if user.is_active:
                login(request, user)
                return render(request, 'profile.html')
    context = {
        "form": form,
    }
    return render(request, 'register.html', context)


# --Views User profiles--
def display_student_profile(request):
    user = User.objects.get(username=request.user.username)
    all_student_info = student_info.objects.filter(user=user)

    for major in all_student_info:
        intended_major = major.Intended_Major
        print intended_major

    context = {'all_student_info': all_student_info}
    return render(request, 'profile.html', context)


# --Edit User Profile(Not working Now)--
def edit_profile(request):
    if request.method == 'POST':
        form = EditProfileForm(request.POST, instance=request.user)

        if form.is_valid():
            form.save()
            return redirect('profile')

    else:
        form = EditProfileForm(instance=request.user)
        args = {'form': form}
        return render(request, 'edit_profile.html', args)



